// use romarin::transpiler::{token::Token, tokenizer};

// #[test]
// fn integer_test() {
//     let mut lex = tokenizer::Lexer::new("5");

//     assert_eq!(lex.yylex().ok(), Some(Token::INT("5".to_string())));
//     assert!(lex.yylex().is_err());
// }

// #[test]
// fn float_test() {
//     let mut lex = tokenizer::Lexer::new("5.1");

//     assert_eq!(lex.yylex().ok(), Some(Token::FLOAT("5.1".to_string())));
//     assert!(lex.yylex().is_err());
// }

// #[test]
// fn id_test() {
//     let mut lex = tokenizer::Lexer::new("x");

//     assert_eq!(lex.yylex().ok(), Some(Token::ID("x".to_string())));
//     assert!(lex.yylex().is_err());
// }

// #[test]
// fn whitespace_test() {
//     let mut lex = tokenizer::Lexer::new(" 5 ");

//     assert_eq!(lex.yylex().ok(), Some(Token::INT("5".to_string())));
//     assert!(lex.yylex().is_err());
// }
