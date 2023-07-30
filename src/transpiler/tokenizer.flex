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

%%
%class Lexer
%result_type Token

%ALPHA=[A-Za-z]
%DIGIT=[0-9]
%WHITE_SPACE_CHAR=[\ \t\b\012]


" "           return self.yylex();
\t            return self.yylex();
\r            return self.yylex();
\n            return self.yylex();
[A-Za-z]([A-Za-z]|[0-9])*   return Ok(ID(self.yytext().to_string()));
[1-9][0-9]*                 return Ok(INT(self.yytext().to_string()));
[1-9][0-9]*\.[0-9]*          return Ok(FLOAT(self.yytext().to_string()));

%%
