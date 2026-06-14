//! Un-move (reverse move) generation
//!
//! This module provides tools for retrograde analysis, allowing you to find
//! predecessor positions and the moves that lead to the current position.
//!
//! Un-move generation follows a callback-based (closure) design: a closure
//! provided by the user is called for each generated pair of predecessor
//! position and associated move.
//!
//! Note that if undoing a move results in a position that might have an
//! active en-passant square, then all cases will be generated (one
//! predecessor with no active en-passant square, plus one or more
//! predecessors with an active en-passant square), since these are
//! distinct predecessor positions.
//! Similarly, multiple predecessors will be generated in case there is the
//! possibility that lost castle rights could be restored in the predecessor
//! position (either on the kingside, on the queenside, or on both sides).
//!
//! Treatment of castling moves can be further controlled through `CastlingRetrogradeMode`.
//!
//! # Examples
//!
//! ```
//! use shakmaty::{Chess, Position, Role, Move, Square};
//! use shakmaty::retrograde::Retrograde;
//! use shakmaty::fen::Epd;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse a successor position from EPD
//! let epd: Epd = "4k3/8/8/8/8/8/R7/4K3 b - -".parse()?;
//! let pos: Chess = epd.into_position(shakmaty::CastlingMode::Standard)?;
//!
//! // Generate legal un-moves
//! let mut unmoves = Vec::new();
//! pos.legal_unmoves(|predecessor, m| {
//!     unmoves.push((predecessor, m));
//! });
//!
//! // We should find the move Ra1-a2
//! assert!(unmoves.iter().any(|(_, m)| {
//!     matches!(m, Move::Normal { role: Role::Rook, from: Square::A1, to: Square::A2, .. })
//! }));
//! # Ok(())
//! # }
//! ```

use crate::{
    attacks, Bitboard, Color, FromSetup, Move, Position, Rank, Role, Setup, Square, Piece,
    EnPassantMode, File, Chess,
};
use crate::setup::EnPassant;
use arrayvec::ArrayVec;

/// The castling mode to use during retrograde analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastlingRetrogradeMode {
    /// Standard chess castling rules.
    Standard,
    /// Chess960 castling rules. Requires the specific starting files of the king and rooks.
    Chess960 {
        /// The starting file of the king.
        king_file: File,
        /// The starting file of the a-side rook.
        a_side_rook_file: File,
        /// The starting file of the h-side rook.
        h_side_rook_file: File,
    },
    /// No castling allowed at all. Useful for endgame tablebase retrograde analysis.
    NoCastling,
}

/// A wrapper struct for configuring and running retrograde analysis on a position.
#[derive(Debug)]
pub struct RetrogradeAnalysis<'a> {
    /// The successor position to analyze.
    pub position: &'a Chess,
    /// The castling mode to use.
    pub castling_mode: CastlingRetrogradeMode,
}

impl<'a> RetrogradeAnalysis<'a> {
    /// Creates a new retrograde analyzer with default configuration (standard castling).
    pub fn new(position: &'a Chess) -> Self {
        Self {
            position,
            castling_mode: CastlingRetrogradeMode::Standard,
        }
    }

    /// Configures the castling mode.
    pub fn with_castling_mode(mut self, castling_mode: CastlingRetrogradeMode) -> Self {
        self.castling_mode = castling_mode;
        self
    }

    fn yield_predecessors<F>(&self, base_setup: Setup, m: Move, f: &mut F)
    where
        F: FnMut(Chess, Move),
    {
        // Multiple generation of predecessors with different castling rights
        // A rook square `sq` can only have its castling right restored if:
        //  - The rook (of the moving side) moved: `m.from() == Some(sq)`
        //  - The rook (of the non-moving side) was captured: `m.to() == sq`
        //  - The king (of the moving side) moved: `m.from() == Some(king)
        let current_rights = if matches!(self.castling_mode, CastlingRetrogradeMode::NoCastling) {
            Bitboard::EMPTY
        } else {
            self.position.castles().castling_rights()
        };

        let mut possible_rights = Bitboard::EMPTY;
        match self.castling_mode {
            CastlingRetrogradeMode::NoCastling => {}
            CastlingRetrogradeMode::Standard => {
                for color in Color::ALL {
                    if let Some(king) = base_setup.board.king_of(color) {
                        let starting_king = color.fold_wb(Square::E1, Square::E8);
                        if king == starting_king {
                            let backrank_rooks = base_setup.board.rooks() & base_setup.board.by_color(color) & color.backrank();
                            let rooks = backrank_rooks & color.fold_wb(
                                Bitboard::from(Square::A1) | Bitboard::from(Square::H1),
                                Bitboard::from(Square::A8) | Bitboard::from(Square::H8),
                            );
                            for sq in rooks {
                                if m.from() == Some(sq) || m.to() == sq || m.from() == Some(king) {
                                    possible_rights.add(sq);
                                }
                            }
                        }
                    }
                }
            }
            CastlingRetrogradeMode::Chess960 { king_file, a_side_rook_file, h_side_rook_file } => {
                for color in Color::ALL {
                    if let Some(king) = base_setup.board.king_of(color) {
                        let starting_king = Square::from_coords(king_file, color.backrank());
                        if king == starting_king {
                            let backrank_rooks = base_setup.board.rooks() & base_setup.board.by_color(color) & color.backrank();
                            let rooks = backrank_rooks & (
                                Bitboard::from(Square::from_coords(a_side_rook_file, color.backrank())) |
                                Bitboard::from(Square::from_coords(h_side_rook_file, color.backrank()))
                            );
                            for sq in rooks {
                                if m.from() == Some(sq) || m.to() == sq || m.from() == Some(king) {
                                    possible_rights.add(sq);
                                }
                            }
                        }
                    }
                }
            }
        }

        let lost_rights = possible_rights & !current_rights;
        let lost_squares: ArrayVec<Square, 4> = lost_rights.into_iter().collect();
        let num_subsets = 1 << lost_squares.len();

        for i in 0..num_subsets {
            let mut extra_rights = Bitboard::EMPTY;
            for (j, &sq) in lost_squares.iter().enumerate() {
                if (i & (1 << j)) != 0 {
                    extra_rights.add(sq);
                }
            }
            let mut setup = base_setup.clone();
            setup.castling_rights = current_rights | extra_rights;

            // Multiple generation of en-passant squares
            // First, try with ep_square = None
            setup.ep_square = None;
            if let Ok(pos) = Chess::from_setup(setup.clone(), self.position.castles().mode()) {
                if let Ok(next_pos) = pos.clone().play(m) {
                    if next_pos == *self.position {
                        f(pos, m);
                    }
                }
            }

            // Next, try with any possible en-passant square
            let inactive_color = self.position.turn();
            let pawns = setup.board.pawns() & setup.board.by_color(inactive_color);
            let relative_4th_rank = inactive_color.relative_rank(Rank::Fourth);
            let eligible_pawns = pawns & Bitboard::from_rank(relative_4th_rank);

            for pawn_sq in eligible_pawns {
                let ep_rank = inactive_color.relative_rank(Rank::Third);
                let ep_sq = Square::from_coords(pawn_sq.file(), ep_rank);
                let pushed_from_rank = inactive_color.relative_rank(Rank::Second);
                let pushed_from = Square::from_coords(pawn_sq.file(), pushed_from_rank);

                if !setup.board.occupied().contains(ep_sq) && !setup.board.occupied().contains(pushed_from) {
                    let mut setup_ep = setup.clone();
                    setup_ep.ep_square = Some(ep_sq);
                    if let Ok(pos) = Chess::from_setup(setup_ep, self.position.castles().mode()) {
                        let mut has_legal_ep = false;
                        for delta in &[-1, 1] {
                            if let Some(from_file) = pawn_sq.file().offset(*delta) {
                                let from_sq = Square::from_coords(from_file, pawn_sq.rank());
                                if pos.board().pawns().contains(from_sq) && pos.board().color_at(from_sq) == Some(pos.turn()) {
                                    let m_ep = Move::EnPassant { from: from_sq, to: ep_sq };
                                    if pos.is_legal(m_ep) {
                                        has_legal_ep = true;
                                        break;
                                    }
                                }
                            }
                        }

                        if has_legal_ep {
                            if let Ok(next_pos) = pos.clone().play(m) {
                                if next_pos == *self.position {
                                    f(pos, m);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Generates all quiet un-moves (no captures, no promotions, no castles).
    pub fn quiet_unmoves<F>(&self, mut f: F)
    where
        F: FnMut(Chess, Move),
    {
        let pos = self.position;
        let us = !pos.turn();
        let them = pos.turn();

        // If successor has an active ep square, only the double pawn push is possible
        if let Some(ep) = pos.maybe_ep_square() {
            let ep_struct = EnPassant(ep);
            let pawn_sq = ep_struct.pawn_pushed_to();
            if pos.board().pawns().contains(pawn_sq) && pos.board().color_at(pawn_sq) == Some(us) {
                let from = ep_struct.pawn_pushed_from();
                if !pos.board().occupied().contains(from) && !pos.board().occupied().contains(ep) {
                    let mut setup = pos.to_setup(EnPassantMode::Always);
                    setup.board.discard_piece_at(pawn_sq);
                    setup.board.set_piece_at(from, Piece { color: us, role: Role::Pawn });
                    setup.turn = us;

                    let m = Move::Normal {
                        role: Role::Pawn,
                        from,
                        to: pawn_sq,
                        capture: None,
                        promotion: None,
                    };

                    self.yield_predecessors(setup, m, &mut f);
                }
            }
            return;
        }

        // Otherwise, generate all quiet un-moves
        for to in pos.board().by_color(us) {
            let piece = pos.board().piece_at(to).unwrap();

            match piece.role {
                Role::Pawn => {
                    let backward_offset = if us == Color::White { -8 } else { 8 };

                    // Single step backward
                    if let Some(from) = to.offset(backward_offset) {
                        if !pos.board().occupied().contains(from) {
                            let mut setup = pos.to_setup(EnPassantMode::Legal);
                            setup.board.discard_piece_at(to);
                            setup.board.set_piece_at(from, Piece { color: us, role: Role::Pawn });
                            setup.turn = us;

                            let m = Move::Normal {
                                role: Role::Pawn,
                                from,
                                to,
                                capture: None,
                                promotion: None,
                            };

                            self.yield_predecessors(setup, m, &mut f);
                        }
                    }

                    // Double step backward
                    let relative_4th = us.fold_wb(Rank::Fourth, Rank::Fifth);
                    if to.rank() == relative_4th {
                        if let Some(from) = to.offset(2 * backward_offset) {
                            let intermediate = to.offset(backward_offset).unwrap();
                            if !pos.board().occupied().contains(from) && !pos.board().occupied().contains(intermediate) {
                                // Check that it does not create any en-passant opportunities
                                let mut has_ep_opportunity = false;
                                if to.file() > File::A {
                                    if let Some(left_piece) = pos.board().piece_at(to.offset(-1).unwrap()) {
                                        if left_piece == (Piece { color: them, role: Role::Pawn }) {
                                            has_ep_opportunity = true;
                                        }
                                    }
                                }
                                if to.file() < File::H {
                                    if let Some(right_piece) = pos.board().piece_at(to.offset(1).unwrap()) {
                                        if right_piece == (Piece { color: them, role: Role::Pawn }) {
                                            has_ep_opportunity = true;
                                        }
                                    }
                                }

                                if !has_ep_opportunity {
                                    let mut setup = pos.to_setup(EnPassantMode::Always);
                                    setup.board.discard_piece_at(to);
                                    setup.board.set_piece_at(from, Piece { color: us, role: Role::Pawn });
                                    setup.turn = us;

                                    let m = Move::Normal {
                                        role: Role::Pawn,
                                        from,
                                        to,
                                        capture: None,
                                        promotion: None,
                                    };

                                    self.yield_predecessors(setup, m, &mut f);
                                }
                            }
                        }
                    }
                }
                _ => {
                    let from_squares = match piece.role {
                        Role::King => attacks::king_attacks(to),
                        Role::Knight => attacks::knight_attacks(to),
                        Role::Bishop => attacks::bishop_attacks(to, pos.board().occupied()),
                        Role::Rook => attacks::rook_attacks(to, pos.board().occupied()),
                        Role::Queen => attacks::queen_attacks(to, pos.board().occupied()),
                        _ => unreachable!(),
                    } & !pos.board().occupied();

                    for from in from_squares {
                        let mut setup = pos.to_setup(EnPassantMode::Always);
                        setup.board.discard_piece_at(to);
                        setup.board.set_piece_at(from, piece);
                        setup.turn = us;

                        let m = Move::Normal {
                            role: piece.role,
                            from,
                            to,
                            capture: None,
                            promotion: None,
                        };

                        self.yield_predecessors(setup, m, &mut f);
                    }
                }
            }
        }
    }

    /// Generates all un-captures of a specific piece role (including en-passant
    /// if the captured piece is a pawn).
    pub fn capture_unmoves<F>(&self, captured_piece: Role, mut f: F)
    where
        F: FnMut(Chess, Move),
    {
        let pos = self.position;
        let us = !pos.turn();
        let them = pos.turn();

        if pos.maybe_ep_square().is_some() {
            return;
        }

        for to in pos.board().by_color(us) {
            let piece = pos.board().piece_at(to).unwrap();

            match piece.role {
                Role::Pawn => {
                    let backward_offset = if us == Color::White { -8 } else { 8 };
                    for file_offset in &[-1, 1] {
                        if let Some(from) = to.offset(backward_offset + file_offset) {
                            if (to.file() == File::A && *file_offset == -1) || (to.file() == File::H && *file_offset == 1) {
                                continue;
                            }
                            if !pos.board().occupied().contains(from) {
                                let mut setup = pos.to_setup(EnPassantMode::Always);
                                setup.board.discard_piece_at(to);
                                setup.board.set_piece_at(from, Piece { color: us, role: Role::Pawn });
                                setup.board.set_piece_at(to, Piece { color: them, role: captured_piece });
                                setup.turn = us;

                                let m = Move::Normal {
                                    role: Role::Pawn,
                                    from,
                                    to,
                                    capture: Some(captured_piece),
                                    promotion: None,
                                };

                                self.yield_predecessors(setup, m, &mut f);
                            }
                        }
                    }
                }
                _ => {
                    let from_squares = match piece.role {
                        Role::King => attacks::king_attacks(to),
                        Role::Knight => attacks::knight_attacks(to),
                        Role::Bishop => attacks::bishop_attacks(to, pos.board().occupied()),
                        Role::Rook => attacks::rook_attacks(to, pos.board().occupied()),
                        Role::Queen => attacks::queen_attacks(to, pos.board().occupied()),
                        _ => unreachable!(),
                    } & !pos.board().occupied();

                    for from in from_squares {
                        let mut setup = pos.to_setup(EnPassantMode::Always);
                        setup.board.discard_piece_at(to);
                        setup.board.set_piece_at(from, piece);
                        setup.board.set_piece_at(to, Piece { color: them, role: captured_piece });
                        setup.turn = us;

                        let m = Move::Normal {
                            role: piece.role,
                            from,
                            to,
                            capture: Some(captured_piece),
                            promotion: None,
                        };

                        self.yield_predecessors(setup, m, &mut f);
                    }
                }
            }
        }

        // En-passant un-captures
        if captured_piece == Role::Pawn {
            let relative_6th = us.fold_wb(Rank::Sixth, Rank::Third);
            for to in pos.board().pawns() & pos.board().by_color(us) & Bitboard::from_rank(relative_6th) {
                let backward_offset = if us == Color::White { -8 } else { 8 };
                for file_offset in &[-1, 1] {
                    if let Some(from) = to.offset(backward_offset + file_offset) {
                        if (to.file() == File::A && *file_offset == -1) || (to.file() == File::H && *file_offset == 1) {
                            continue;
                        }
                        let captured_sq = to.offset(backward_offset).unwrap();
                        if !pos.board().occupied().contains(from) && !pos.board().occupied().contains(captured_sq) {
                            let mut setup = pos.to_setup(EnPassantMode::Always);
                            setup.board.discard_piece_at(to);
                            setup.board.set_piece_at(from, Piece { color: us, role: Role::Pawn });
                            setup.board.set_piece_at(captured_sq, Piece { color: them, role: Role::Pawn });
                            setup.turn = us;
                            setup.ep_square = Some(to);

                            let m = Move::EnPassant { from, to };

                            if let Ok(pos_ep) = Chess::from_setup(setup, pos.castles().mode()) {
                                if let Ok(next_pos) = pos_ep.clone().play(m) {
                                    if next_pos == *pos {
                                        f(pos_ep, m);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Generates all un-promotions of a piece on the last rank back to a pawn (with no capture).
    pub fn promotion_unmoves<F>(&self, mut f: F)
    where
        F: FnMut(Chess, Move),
    {
        let pos = self.position;
        let us = !pos.turn();

        if pos.maybe_ep_square().is_some() {
            return;
        }

        let relative_8th = us.fold_wb(Rank::Eighth, Rank::First);
        for to in pos.board().by_color(us) & Bitboard::from_rank(relative_8th) {
            let piece = pos.board().piece_at(to).unwrap();
            if piece.role == Role::Pawn || piece.role == Role::King {
                continue;
            }

            let backward_offset = if us == Color::White { -8 } else { 8 };
            if let Some(from) = to.offset(backward_offset) {
                if !pos.board().occupied().contains(from) {
                    let mut setup = pos.to_setup(EnPassantMode::Always);
                    setup.board.discard_piece_at(to);
                    setup.board.set_piece_at(from, Piece { color: us, role: Role::Pawn });
                    setup.turn = us;

                    let m = Move::Normal {
                        role: Role::Pawn,
                        from,
                        to,
                        capture: None,
                        promotion: Some(piece.role),
                    };

                    self.yield_predecessors(setup, m, &mut f);
                }
            }
        }
    }

    /// Generates all un-promotions of a piece on the last rank back to a pawn with an un-capture.
    pub fn promotion_capture_unmoves<F>(&self, captured_piece: Role, mut f: F)
    where
        F: FnMut(Chess, Move),
    {
        let pos = self.position;
        let us = !pos.turn();
        let them = pos.turn();

        if pos.maybe_ep_square().is_some() {
            return;
        }

        let relative_8th = us.fold_wb(Rank::Eighth, Rank::First);
        for to in pos.board().by_color(us) & Bitboard::from_rank(relative_8th) {
            let piece = pos.board().piece_at(to).unwrap();
            if piece.role == Role::Pawn || piece.role == Role::King {
                continue;
            }

            let backward_offset = if us == Color::White { -8 } else { 8 };
            for file_offset in &[-1, 1] {
                if let Some(from) = to.offset(backward_offset + file_offset) {
                    if (to.file() == File::A && *file_offset == -1) || (to.file() == File::H && *file_offset == 1) {
                        continue;
                    }
                    if !pos.board().occupied().contains(from) {
                        let mut setup = pos.to_setup(EnPassantMode::Always);
                        setup.board.discard_piece_at(to);
                        setup.board.set_piece_at(from, Piece { color: us, role: Role::Pawn });
                        setup.board.set_piece_at(to, Piece { color: them, role: captured_piece });
                        setup.turn = us;

                        let m = Move::Normal {
                            role: Role::Pawn,
                            from,
                            to,
                            capture: Some(captured_piece),
                            promotion: Some(piece.role),
                        };

                        self.yield_predecessors(setup, m, &mut f);
                    }
                }
            }
        }
    }

    /// Generates all castling reversals.
    pub fn castling_unmoves<F>(&self, mut f: F)
    where
        F: FnMut(Chess, Move),
    {
        let us = !self.position.turn();

        if self.position.maybe_ep_square().is_some() {
            return;
        }

        let (king_file, a_side_rook_file, h_side_rook_file) = match self.castling_mode {
            CastlingRetrogradeMode::NoCastling => return,
            CastlingRetrogradeMode::Standard => (File::E, File::A, File::H),
            CastlingRetrogradeMode::Chess960 { king_file, a_side_rook_file, h_side_rook_file } => {
                (king_file, a_side_rook_file, h_side_rook_file)
            }
        };

        let backrank = us.backrank();

        // King-side (h-side) castling
        let king_to = Square::from_coords(File::G, backrank);
        let rook_to = Square::from_coords(File::F, backrank);
        let king_from = Square::from_coords(king_file, backrank);
        let rook_from = Square::from_coords(h_side_rook_file, backrank);

        if self.position.board().king_of(us) == Some(king_to)
            && self.position.board().rooks().contains(rook_to)
            && self.position.board().color_at(rook_to) == Some(us)
        {
            if !self.position.board().occupied().without(king_to).without(rook_to).contains(king_from)
                && !self.position.board().occupied().without(king_to).without(rook_to).contains(rook_from)
            {
                let mut setup = self.position.to_setup(EnPassantMode::Always);
                setup.board.discard_piece_at(king_to);
                setup.board.discard_piece_at(rook_to);
                setup.board.set_piece_at(king_from, Piece { color: us, role: Role::King });
                setup.board.set_piece_at(rook_from, Piece { color: us, role: Role::Rook });
                setup.turn = us;

                let m = Move::Castle { king: king_from, rook: rook_from };

                self.yield_predecessors(setup, m, &mut f);
            }
        }

        // Queen-side (a-side) castling
        let king_to = Square::from_coords(File::C, backrank);
        let rook_to = Square::from_coords(File::D, backrank);
        let king_from = Square::from_coords(king_file, backrank);
        let rook_from = Square::from_coords(a_side_rook_file, backrank);

        if self.position.board().king_of(us) == Some(king_to)
            && self.position.board().rooks().contains(rook_to)
            && self.position.board().color_at(rook_to) == Some(us)
        {
            if !self.position.board().occupied().without(king_to).without(rook_to).contains(king_from)
                && !self.position.board().occupied().without(king_to).without(rook_to).contains(rook_from)
            {
                let mut setup = self.position.to_setup(EnPassantMode::Always);
                setup.board.discard_piece_at(king_to);
                setup.board.discard_piece_at(rook_to);
                setup.board.set_piece_at(king_from, Piece { color: us, role: Role::King });
                setup.board.set_piece_at(rook_from, Piece { color: us, role: Role::Rook });
                setup.turn = us;

                let m = Move::Castle { king: king_from, rook: rook_from };

                self.yield_predecessors(setup, m, &mut f);
            }
        }
    }

    /// Generates all legal un-moves.
    pub fn legal_unmoves<F>(&self, mut f: F)
    where
        F: FnMut(Chess, Move),
    {
        self.quiet_unmoves(&mut f);
        for &role in &[Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
            self.capture_unmoves(role, &mut f);
        }
        self.promotion_unmoves(&mut f);
        for &role in &[Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
            self.promotion_capture_unmoves(role, &mut f);
        }
        self.castling_unmoves(&mut f);
    }
}

pub trait Retrograde: Position + FromSetup + Clone {
    /// Generates all quiet un-moves (no captures, no promotions, no castles).
    fn quiet_unmoves<F>(&self, f: F)
    where
        F: FnMut(Self, Move);

    /// Generates all un-captures of a specific piece role (including en-passant
    /// if the captured piece is a pawn).
    fn capture_unmoves<F>(&self, captured_piece: Role, f: F)
    where
        F: FnMut(Self, Move);

    /// Generates all un-promotions of a piece on the last rank back to a pawn (with no capture).
    fn promotion_unmoves<F>(&self, f: F)
    where
        F: FnMut(Self, Move);

    /// Generates all un-promotions of a piece on the last rank back to a pawn with an un-capture.
    fn promotion_capture_unmoves<F>(&self, captured_piece: Role, f: F)
    where
        F: FnMut(Self, Move);

    /// Generates all castling reversals.
    fn castling_unmoves<F>(&self, f: F)
    where
        F: FnMut(Self, Move);

    /// Generates all legal un-moves.
    fn legal_unmoves<F>(&self, f: F)
    where
        F: FnMut(Self, Move);
}

impl Retrograde for Chess {
    fn quiet_unmoves<F>(&self, f: F)
    where
        F: FnMut(Self, Move),
    {
        RetrogradeAnalysis::new(self).quiet_unmoves(f);
    }

    fn capture_unmoves<F>(&self, captured_piece: Role, f: F)
    where
        F: FnMut(Self, Move),
    {
        RetrogradeAnalysis::new(self).capture_unmoves(captured_piece, f);
    }

    fn promotion_unmoves<F>(&self, f: F)
    where
        F: FnMut(Self, Move),
    {
        RetrogradeAnalysis::new(self).promotion_unmoves(f);
    }

    fn promotion_capture_unmoves<F>(&self, captured_piece: Role, f: F)
    where
        F: FnMut(Self, Move),
    {
        RetrogradeAnalysis::new(self).promotion_capture_unmoves(captured_piece, f);
    }

    fn castling_unmoves<F>(&self, f: F)
    where
        F: FnMut(Self, Move),
    {
        RetrogradeAnalysis::new(self).castling_unmoves(f);
    }

    fn legal_unmoves<F>(&self, f: F)
    where
        F: FnMut(Self, Move),
    {
        RetrogradeAnalysis::new(self).legal_unmoves(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::Fen;
    use crate::CastlingMode;
    use alloc::vec::Vec;

    fn setup_pos(fen_str: &str) -> Chess {
        fen_str
            .parse::<Fen>()
            .expect("valid fen")
            .into_position::<Chess>(CastlingMode::Standard)
            .expect("legal position")
    }

    #[test]
    fn test_quiet_unmoves() {
        // White King on e1, Black King on e8, White Rook on a2.
        // It is Black's turn.
        // Predecessor moves by White: e.g. Ra1-a2.
        let pos = setup_pos("4k3/8/8/8/8/8/R7/4K3 b - - 0 1");
        let mut unmoves = Vec::new();
        pos.quiet_unmoves(|p, m| unmoves.push((p, m)));

        // There should be a move Ra1-a2
        let found = unmoves.iter().any(|(_, m)| {
            matches!(m, Move::Normal { role: Role::Rook, from: Square::A1, to: Square::A2, .. })
        });
        assert!(found, "Should find Ra1-a2 quiet unmove");
    }

    #[test]
    fn test_double_step_pawn_unmove() {
        // White pawn on e4, White King on e1, Black King on e8.
        // It is Black's turn.
        // If ep_square is None, e2-e4 double step backward is possible because there are no adjacent Black pawns.
        let pos = setup_pos("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1");
        let mut unmoves = Vec::new();
        pos.quiet_unmoves(|p, m| unmoves.push((p, m)));

        let found_double = unmoves.iter().any(|(_, m)| {
            matches!(m, Move::Normal { role: Role::Pawn, from: Square::E2, to: Square::E4, .. })
        });
        assert!(found_double, "Should find e2-e4 double step backward when no adjacent pawns exist");

        // If there is an adjacent Black pawn on d4, double step backward should NOT be possible
        // because it would have created an en-passant opportunity (which is absent in the successor).
        let pos_with_adj = setup_pos("4k3/8/8/3pP3/8/8/8/4K3 b - - 0 1");
        let mut unmoves_adj = Vec::new();
        pos_with_adj.quiet_unmoves(|p, m| unmoves_adj.push((p, m)));

        let found_double_adj = unmoves_adj.iter().any(|(_, m)| {
            matches!(m, Move::Normal { role: Role::Pawn, from: Square::E2, to: Square::E5, .. })
        });
        assert!(!found_double_adj, "Should NOT find e2-e4 double step backward when adjacent pawn exists");
    }

    #[test]
    fn test_capture_unmoves() {
        // White King on e1, Black King on e8, White Bishop on d4.
        // It is Black's turn.
        // We want to un-capture a Black Knight on d4.
        let pos = setup_pos("4k3/8/8/8/3B4/8/8/4K3 b - - 0 1");
        let mut unmoves = Vec::new();
        pos.capture_unmoves(Role::Knight, |p, m| unmoves.push((p, m)));

        // There should be a move e3xd4 or similar
        let found = unmoves.iter().any(|(p, m)| {
            if let Move::Normal { role: Role::Bishop, from: _, to: Square::D4, capture: Some(Role::Knight), .. } = m {
                p.board().piece_at(Square::D4) == Some(Piece { color: Color::Black, role: Role::Knight })
            } else {
                false
            }
        });
        assert!(found, "Should find Bishop un-capturing a Knight on d4");
    }

    #[test]
    fn test_en_passant_uncapture() {
        // White pawn on d6, White King on e1, Black King on e8.
        // It is Black's turn.
        // We want to un-capture a Black pawn en-passant.
        // This means White pawn on d6 came from e5 (or c5) and captured Black pawn on d5 en-passant.
        let pos = setup_pos("4k3/8/3P4/8/8/8/8/4K3 b - - 0 1");
        let mut unmoves = Vec::new();
        pos.capture_unmoves(Role::Pawn, |p, m| unmoves.push((p, m)));

        let found = unmoves.iter().any(|(p, m)| {
            if let Move::EnPassant { from: Square::E5, to: Square::D6 } = m {
                p.board().piece_at(Square::E5) == Some(Piece { color: Color::White, role: Role::Pawn })
                    && p.board().piece_at(Square::D5) == Some(Piece { color: Color::Black, role: Role::Pawn })
                    && p.maybe_ep_square() == Some(Square::D6)
            } else {
                false
            }
        });
        assert!(found, "Should find exd6 en-passant un-capture");
    }

    #[test]
    fn test_promotion_unmoves() {
        // White Queen on e8, White King on e1, Black King on a8.
        // It is Black's turn.
        // We want to reverse a quiet promotion.
        // This means White Queen on e8 came from e7 as a pawn and promoted.
        let pos = setup_pos("k3Q3/8/8/8/8/8/8/4K3 b - - 0 1");
        let mut unmoves = Vec::new();
        pos.promotion_unmoves(|p, m| unmoves.push((p, m)));

        let found = unmoves.iter().any(|(p, m)| {
            if let Move::Normal { role: Role::Pawn, from: Square::E7, to: Square::E8, capture: Option::None, promotion: Some(Role::Queen) } = m {
                p.board().piece_at(Square::E7) == Some(Piece { color: Color::White, role: Role::Pawn })
            } else {
                false
            }
        });
        assert!(found, "Should find e7-e8=Q quiet un-promotion");
    }

    #[test]
    fn test_castling_unmoves() {
        // White King on g1, White Rook on f1, Black King on e8.
        // It is Black's turn.
        // We want to reverse King-side castling.
        let pos = setup_pos("4k3/8/8/8/8/8/8/5RK1 b - - 0 1");
        let mut unmoves = Vec::new();
        pos.castling_unmoves(|p, m| unmoves.push((p, m)));

        let found = unmoves.iter().any(|(p, m)| {
            if let Move::Castle { king: Square::E1, rook: Square::H1 } = m {
                p.board().king_of(Color::White) == Some(Square::E1)
                    && p.board().rooks().contains(Square::H1)
            } else {
                false
            }
        });
        assert!(found, "Should find O-O castling reversal");
    }

    #[test]
    fn test_double_generation_ep() {
        // White Rook on a2, White King on e2, Black King on e8.
        // White Pawn on d5, Black pawn on e5.
        // It is Black's turn.
        // Since the King is on e2, castling is structurally impossible, so we isolate ep double-generation.
        // For the quiet unmove Ra1-a2, we should get two predecessor positions:
        // 1. One with ep_square = None
        // 2. One with ep_square = Some(e6) (since Black has a pawn on e5 and White has an adjacent pawn on d5)
        let pos = setup_pos("4k3/8/8/3Pp3/8/8/R3K3/8 b - - 0 1");
        let mut unmoves = Vec::new();
        pos.quiet_unmoves(|p, m| unmoves.push((p, m)));

        let ep_none_count = unmoves.iter().filter(|(p, m)| {
            matches!(m, Move::Normal { role: Role::Rook, from: Square::A1, to: Square::A2, .. })
                && p.maybe_ep_square().is_none()
        }).count();

        let ep_some_count = unmoves.iter().filter(|(p, m)| {
            matches!(m, Move::Normal { role: Role::Rook, from: Square::A1, to: Square::A2, .. })
                && p.maybe_ep_square() == Some(Square::E6)
        }).count();

        assert_eq!(ep_none_count, 1, "Should find exactly one predecessor with ep_square = None");
        assert_eq!(ep_some_count, 1, "Should find exactly one predecessor with ep_square = Some(e6)");
    }

    #[test]
    fn test_double_generation_castling() {
        // White King on e2, Black King on e8.
        // White Rooks on a1 and h1.
        // It is Black's turn.
        // For the quiet unmove Ke1-e2, we should get multiple predecessor positions
        // with different castling rights (since the King on e1 and Rooks on a1/h1 allow castling rights to be restored).
        let pos = setup_pos("4k3/8/8/8/8/8/4K3/R6R b - - 0 1");
        let mut unmoves = Vec::new();
        pos.quiet_unmoves(|p, m| unmoves.push((p, m)));

        let has_both = unmoves.iter().any(|(p, m)| {
            matches!(m, Move::Normal { role: Role::King, from: Square::E1, to: Square::E2, .. })
                && p.castles().has(Color::White, crate::CastlingSide::KingSide)
                && p.castles().has(Color::White, crate::CastlingSide::QueenSide)
        });

        let has_none = unmoves.iter().any(|(p, m)| {
            matches!(m, Move::Normal { role: Role::King, from: Square::E1, to: Square::E2, .. })
                && !p.castles().has(Color::White, crate::CastlingSide::KingSide)
                && !p.castles().has(Color::White, crate::CastlingSide::QueenSide)
        });

        assert!(has_both, "Should find predecessor with both castling rights restored");
        assert!(has_none, "Should find predecessor with no castling rights restored");
    }

    #[test]
    fn test_en_passant_legality_predecessor() {
        // White King on a5, White Pawn on b5, Black Pawn on c5, Black Rook on h5, Black King on e8.
        // It is Black's turn.
        // If Black's last move was Rg5-h5, then in the predecessor, the Black Rook was on g5.
        // If we try to set ep_square = Some(c6), the en-passant capture b5xc6 is illegal
        // because of the horizontal pin along the 5th rank (King on a5, Pawn on b5, Pawn on c5, Rook on g5).
        // So the predecessor with ep_square = Some(c6) should NOT be generated.
        let pos = setup_pos("4k3/8/8/KPp4r/8/8/8/8 b - - 0 1");
        let mut unmoves = Vec::new();
        pos.quiet_unmoves(|p, m| unmoves.push((p, m)));

        let ep_some_count = unmoves.iter().filter(|(p, m)| {
            matches!(m, Move::Normal { role: Role::Rook, from: Square::G5, to: Square::H5, .. })
                && p.maybe_ep_square() == Some(Square::C6)
        }).count();

        assert_eq!(ep_some_count, 0, "Should NOT generate predecessor with ep_square = Some(c6) due to illegal capture");
    }

    #[test]
    fn test_retrograde_analysis_no_castling() {
        // White King on e2, Black King on e8.
        // White Rooks on a1 and h1.
        // It is Black's turn.
        // If we use NoCastling mode, no castling rights should be restored, and no castling unmoves should be generated.
        let pos = setup_pos("4k3/8/8/8/8/8/4K3/R6R b - - 0 1");
        let mut unmoves = Vec::new();
        RetrogradeAnalysis::new(&pos)
            .with_castling_mode(CastlingRetrogradeMode::NoCastling)
            .quiet_unmoves(|p, m| unmoves.push((p, m)));

        for (p, _) in &unmoves {
            assert!(!p.castles().has(Color::White, crate::CastlingSide::KingSide));
            assert!(!p.castles().has(Color::White, crate::CastlingSide::QueenSide));
        }
    }

    #[test]
    fn test_retrograde_analysis_chess960() {
        // Let's test Chess960 castling unmoves with custom starting files.
        // Suppose King starts on f1, left Rook on c1, right Rook on g1.
        // White King on g1 (king_to), White Rook on f1 (rook_to).
        // It is Black's turn.
        // If we reverse castling under Chess960 with king_file = F, h_side_rook_file = G:
        // The king should be restored to f1 and rook to g1.
        let pos = "4k3/8/8/8/8/8/8/5RK1 b - - 0 1"
            .parse::<Fen>()
            .expect("valid fen")
            .into_position::<Chess>(CastlingMode::Chess960)
            .expect("legal position");
        let mut unmoves = Vec::new();
        RetrogradeAnalysis::new(&pos)
            .with_castling_mode(CastlingRetrogradeMode::Chess960 {
                king_file: File::F,
                a_side_rook_file: File::C,
                h_side_rook_file: File::G,
            })
            .castling_unmoves(|p, m| unmoves.push((p, m)));

        let found = unmoves.iter().any(|(p, m)| {
            if let Move::Castle { king: Square::F1, rook: Square::G1 } = m {
                p.board().king_of(Color::White) == Some(Square::F1)
                    && p.board().rooks().contains(Square::G1)
            } else {
                false
            }
        });
        assert!(found, "Should find Chess960 castling reversal with custom starting files");
    }
}
