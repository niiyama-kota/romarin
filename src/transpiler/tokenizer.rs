use super::token::Token;
use Token::*;

#[test]
fn integer_test() {
    let mut lex = Lexer::new("5");

    assert_eq!(lex.yylex().ok(), Some(Token::INT("5".to_string())));
    assert!(lex.yylex().is_err());
}

#[test]
fn float_test() {
    let mut lex = Lexer::new("5.1");

    assert_eq!(lex.yylex().ok(), Some(Token::FLOAT("5.1".to_string())));
    assert!(lex.yylex().is_err());
}

#[test]
fn id_test() {
    let mut lex = Lexer::new("x");

    assert_eq!(lex.yylex().ok(), Some(Token::ID("x".to_string())));
    assert!(lex.yylex().is_err());
}

#[test]
fn whitespace_test() {
    let mut lex = Lexer::new(" 5 ");

    assert_eq!(lex.yylex().ok(), Some(Token::INT("5".to_string())));
    assert!(lex.yylex().is_err());
}

use std::collections::HashMap;
use std::str::CharIndices;

#[derive(Debug, PartialEq)]
pub enum Error {
    EOF,
    Unmatch,
}

pub struct Lexer<'a> {
    input: &'a str,
    cmap: Vec<usize>,
    cmap2: HashMap<usize, usize>,
    start: CharIndices<'a>,
    current: CharIndices<'a>,
    max_len: usize,

    zz_state: usize,
    zz_lexical_state: usize,

    // byte
    zz_marked_pos: usize,
    zz_current_pos: usize,
    zz_start_read: usize,

    // char
    zz_start_read_char: usize,
    zz_marked_char: usize,

    zz_at_eof: bool,
}

impl<'a> Lexer<'a> {
    pub const ZZ_ROW: [usize; 5] = [0, 5, 10, 15, 20];
    pub const ZZ_TRANS: [i32; 25] = [
        -1, 1, 2, -1, 3, -1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 4, -1, -1, 3, 3, -1, -1, -1, 4, 4,
    ];
    pub const ZZ_ATTR: [i32; 5] = [0, 9, 1, 1, 1];
    pub const ZZ_ACTION: [i32; 5] = [0, 1, 2, 3, 4];
    pub const ZZ_LEXSTATE: [i32; 2] = [0, 0];
    pub const YYINITIAL: usize = 0;

    pub const YYEOF: i32 = -1;

    pub fn new(input: &'a str) -> Lexer<'a> {
        let max_len = input.chars().clone().count();
        let mut cmap: Vec<usize> = Vec::with_capacity(256);
        cmap.resize(256, 0);
        let mut cmap2: HashMap<usize, usize> = HashMap::new();
        cmap[9] = 1;
        cmap[10] = 1;
        cmap[11] = 1;
        cmap[12] = 1;
        cmap[13] = 1;
        cmap[32] = 1;
        cmap[48] = 3;
        cmap[49] = 4;
        cmap[50] = 4;
        cmap[51] = 4;
        cmap[52] = 4;
        cmap[53] = 4;
        cmap[54] = 4;
        cmap[55] = 4;
        cmap[56] = 4;
        cmap[57] = 4;
        cmap[65] = 2;
        cmap[66] = 2;
        cmap[67] = 2;
        cmap[68] = 2;
        cmap[69] = 2;
        cmap[70] = 2;
        cmap[71] = 2;
        cmap[72] = 2;
        cmap[73] = 2;
        cmap[74] = 2;
        cmap[75] = 2;
        cmap[76] = 2;
        cmap[77] = 2;
        cmap[78] = 2;
        cmap[79] = 2;
        cmap[80] = 2;
        cmap[81] = 2;
        cmap[82] = 2;
        cmap[83] = 2;
        cmap[84] = 2;
        cmap[85] = 2;
        cmap[86] = 2;
        cmap[87] = 2;
        cmap[88] = 2;
        cmap[89] = 2;
        cmap[90] = 2;
        cmap[97] = 2;
        cmap[98] = 2;
        cmap[99] = 2;
        cmap[100] = 2;
        cmap[101] = 2;
        cmap[102] = 2;
        cmap[103] = 2;
        cmap[104] = 2;
        cmap[105] = 2;
        cmap[106] = 2;
        cmap[107] = 2;
        cmap[108] = 2;
        cmap[109] = 2;
        cmap[110] = 2;
        cmap[111] = 2;
        cmap[112] = 2;
        cmap[113] = 2;
        cmap[114] = 2;
        cmap[115] = 2;
        cmap[116] = 2;
        cmap[117] = 2;
        cmap[118] = 2;
        cmap[119] = 2;
        cmap[120] = 2;
        cmap[121] = 2;
        cmap[122] = 2;
        cmap[133] = 1;
        cmap2.insert(8232, 1);
        cmap2.insert(8233, 1);

        Lexer {
            input,
            cmap,
            cmap2,
            start: input.char_indices(),
            current: input.char_indices(),

            max_len,
            zz_state: 0,
            zz_lexical_state: Lexer::YYINITIAL,
            zz_marked_pos: 0,
            zz_current_pos: 0,
            zz_start_read: 0,
            zz_start_read_char: 0,
            zz_marked_char: 0,

            zz_at_eof: false,
        }
    }

    pub fn is_eof(&self) -> bool {
        self.zz_at_eof
    }

    pub fn yybegin(&mut self, new_state: usize) {
        self.zz_lexical_state = new_state;
    }

    pub fn yystate(&self) -> usize {
        self.zz_lexical_state
    }

    pub fn yylength(&self) -> usize {
        self.zz_marked_char - self.zz_start_read_char
    }

    pub fn yycharat(&self, pos: usize) -> Option<char> {
        let mut ch: Option<char> = None;
        let mut start = self.start.clone();
        for _ in 0..(pos + 1) {
            if let Some(c) = start.next() {
                ch = Some(c.1);
            } else {
                return None;
            }
        }
        ch
    }

    pub fn yytext(&self) -> String {
        self.input[self.yybytepos()].to_string()
    }

    pub fn yytextpos(&self) -> std::ops::Range<usize> {
        std::ops::Range {
            start: self.zz_start_read_char,
            end: self.zz_marked_char,
        }
    }

    pub fn yybytepos(&self) -> std::ops::Range<usize> {
        std::ops::Range {
            start: self.zz_start_read,
            end: self.zz_marked_pos,
        }
    }

    pub fn yylex(&mut self) -> Result<Token, Error> {
        let mut zz_input: i32 = -1;

        // cached
        loop {
            // char unit
            let mut zz_marked_char_l = self.zz_marked_char;
            let mut zz_current_char_pos_l = self.zz_marked_char;
            self.zz_start_read_char = self.zz_marked_char;

            // byte unit
            let mut zz_marked_byte_pos_l = self.zz_marked_pos;
            let mut zz_current_byte_pos_l = self.zz_marked_pos;

            let mut zz_action = -1;
            let mut current = self.current.clone();

            self.zz_start_read = self.zz_marked_pos;
            self.zz_current_pos = self.zz_marked_pos;
            self.zz_start_read_char = self.zz_marked_char;
            self.start = self.current.clone();

            self.zz_state = Lexer::ZZ_LEXSTATE[self.zz_lexical_state] as usize;

            // set up zz_action for empty match case:
            let zz_attributes = Lexer::ZZ_ATTR[self.zz_state];
            if (zz_attributes & 1) == 1 {
                zz_action = self.zz_state as i32;
            }

            'zz_for_action: loop {
                if zz_current_char_pos_l < self.max_len {
                    if let Some(next) = current.next() {
                        zz_current_byte_pos_l += next.1.len_utf8();
                        zz_input = next.1 as i32;
                    }
                    zz_current_char_pos_l += 1;
                } else if self.zz_at_eof {
                    zz_input = Lexer::YYEOF;
                    break 'zz_for_action;
                } else {
                    self.zz_current_pos = zz_current_byte_pos_l;

                    if self.max_len <= zz_current_char_pos_l {
                        zz_input = Lexer::YYEOF;
                        break 'zz_for_action;
                    } else {
                        if let Some(next) = current.next() {
                            zz_current_byte_pos_l += next.1.len_utf8();
                            zz_input = next.1 as i32;
                        }
                        zz_current_char_pos_l += 1;
                    }
                }

                let cidx = if zz_input <= 0xFF {
                    self.cmap[zz_input as usize]
                } else {
                    *self.cmap2.get(&(zz_input as usize)).unwrap_or(&0usize)
                };
                let idx = Lexer::ZZ_ROW[self.zz_state] + cidx;
                let zz_next = Lexer::ZZ_TRANS[idx];
                if zz_next == -1 {
                    break 'zz_for_action;
                }
                self.zz_state = zz_next as usize;

                let zz_attributes = Lexer::ZZ_ATTR[self.zz_state];
                if (zz_attributes & 1) == 1 {
                    zz_action = self.zz_state as i32;
                    zz_marked_char_l = zz_current_char_pos_l;
                    zz_marked_byte_pos_l = zz_current_byte_pos_l;
                    self.current = current.clone();

                    if (zz_attributes & 8) == 8 {
                        break 'zz_for_action;
                    }
                }
            } // loop 'zz_for_action

            // store back cached position
            self.zz_marked_char = zz_marked_char_l;
            self.zz_marked_pos = zz_marked_byte_pos_l;

            if zz_input == Lexer::YYEOF && self.zz_start_read == self.zz_current_pos {
                self.zz_at_eof = true;

                return Err(Error::EOF);
            } else {
                let action = if zz_action < 0 {
                    zz_action
                } else {
                    Lexer::ZZ_ACTION[zz_action as usize]
                };
                match action {
                    1 => {
                        return self.yylex();
                    }
                    5 => { /* nothing */ }
                    2 => {
                        return Ok(ID(self.yytext().to_string()));
                    }
                    6 => { /* nothing */ }
                    3 => {
                        return Ok(INT(self.yytext().to_string()));
                    }
                    7 => { /* nothing */ }
                    4 => {
                        return Ok(FLOAT(self.yytext().to_string()));
                    }
                    8 => { /* nothing */ }

                    _ => {
                        return Err(Error::Unmatch);
                    }
                }
            }
        } // loop
          // never reach end of function
    }
}
