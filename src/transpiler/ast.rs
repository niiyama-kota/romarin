use std::default;

use super::{node::Node, tokenizer::Lexer};
use crate::transpiler::token::Token;

#[derive(Debug)]
pub struct Ast {
    root: Node,
    children: Option<Vec<Box<Ast>>>,
}

impl Ast {
    fn new(_root: Node) -> Self {
        Ast {
            root: _root,
            children: None,
        }
    }
    fn add_child(mut self, child: Ast) {
        match self.children {
            Some(mut children) => children.push(Box::new(child)),
            None => self.children = Some(vec![Box::new(child)]),
        }
    }
}

fn program(input: &mut Lexer<'_>) -> Vec<Ast> {
    let mut statements: Vec<Ast> = Vec::<Ast>::default();

    while !input.is_eof() {
        match literal(input) {
            Some(statement) => statements.push(statement),
            None => (),
        }
    }

    return statements;
}

fn literal(input: &mut Lexer<'_>) -> Option<Ast> {
    if let Some(token) = input.yylex().ok() {
        match token {
            Token::ID(identifier) => {
                return Some(Ast::new(Node::ID(identifier)));
            }
            Token::INT(num) => return Some(Ast::new(Node::INT(num.parse::<i32>().unwrap()))),
            Token::FLOAT(num) => return Some(Ast::new(Node::FLOAT(num.parse::<f32>().unwrap()))),
        }
    } else {
        return None;
    }
}

#[test]
fn construct_literal_ast_test() {
    let mut input = Lexer::new("5.1");
    let t = program(&mut input);

    println!("{:?}", t);
}
