use super::{ast::Ast, token::Token, tokenizer::Lexer};

pub fn to_verilog(mut program: Ast) {
    // let tokens: Vec<Token> = tokenize(input);
    // let ast = parse(&tokens);

    let identifier = "MLDeviceBehavior";
    println!("function real {};", identifier);
    println!("  input d, g, s;");
    println!("  electrical d, g, s;");
    println!("  real Id, Vgs, Vds;");
    println!("  begin");
    println!("      Vgs = V(g, s);");
    println!("      Vds = V(d, s);");
    //codegen(program, tab = "\t\t");
    println!("      MLDeviceBehavior = Id;");
    println!("  end");
    println!("endfunction");
}
