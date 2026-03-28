//! calc - Command-line expression calculator.
//!
//! A simple interactive calculator that evaluates arithmetic expressions.
//! Supports variables, common math functions (sin, cos, tan, log, exp, sqrt,
//! abs, min, max), and both degree and radian modes.
//!
//! Translated from IMOD's calc.f (DESKCALC by D. Kopf, 1985).

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::process;

use clap::Parser;

#[derive(Parser)]
#[command(name = "calc", about = "Command-line expression calculator")]
struct Args {
    /// Expression to evaluate (if given, evaluate and exit)
    expression: Option<String>,

    /// Use degrees for trigonometric functions
    #[arg(long)]
    degrees: bool,
}

struct CalcState {
    vars: HashMap<String, f64>,
    use_degrees: bool,
}

impl CalcState {
    fn new(use_degrees: bool) -> Self {
        let mut vars = HashMap::new();
        vars.insert("PI".to_string(), std::f64::consts::PI);
        vars.insert("pi".to_string(), std::f64::consts::PI);
        vars.insert("E".to_string(), std::f64::consts::E);
        vars.insert("e".to_string(), std::f64::consts::E);
        CalcState { vars, use_degrees }
    }

    fn angle_to_rad(&self, x: f64) -> f64 {
        if self.use_degrees {
            x.to_radians()
        } else {
            x
        }
    }

    fn rad_to_angle(&self, x: f64) -> f64 {
        if self.use_degrees {
            x.to_degrees()
        } else {
            x
        }
    }

    fn evaluate(&mut self, expr: &str) -> Result<f64, String> {
        let expr = expr.trim();
        if expr.is_empty() {
            return Err("empty expression".to_string());
        }

        // Check for assignment: VAR = expr
        if let Some(eq_pos) = expr.find('=') {
            let var_name = expr[..eq_pos].trim().to_string();
            if var_name.is_empty()
                || !var_name
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '#')
            {
                return Err(format!("invalid variable name: {}", var_name));
            }
            let val = self.eval_expr(&expr[eq_pos + 1..])?;
            self.vars.insert(var_name, val);
            return Ok(val);
        }

        self.eval_expr(expr)
    }

    fn eval_expr(&self, expr: &str) -> Result<f64, String> {
        let expr = expr.trim();
        // Simple recursive descent parser
        let tokens = tokenize(expr)?;
        let mut pos = 0;
        let result = parse_add_sub(&tokens, &mut pos, self)?;
        if pos < tokens.len() {
            return Err(format!("unexpected token at position {}", pos));
        }
        Ok(result)
    }
}

#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
    Comma,
}

fn tokenize(expr: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' => i += 1,
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    tokens.push(Token::Caret);
                    i += 2;
                } else {
                    tokens.push(Token::Star);
                    i += 1;
                }
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '%' => {
                tokens.push(Token::Percent);
                i += 1;
            }
            '^' => {
                tokens.push(Token::Caret);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                // Handle scientific notation
                if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                    i += 1;
                    if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                        i += 1;
                    }
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let s: String = chars[start..i].iter().collect();
                let val = s
                    .parse::<f64>()
                    .map_err(|_| format!("invalid number: {}", s))?;
                tokens.push(Token::Number(val));
            }
            c if c.is_ascii_alphabetic() || c == '_' || c == '#' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '#')
                {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(s));
            }
            c => return Err(format!("unexpected character: {}", c)),
        }
    }
    Ok(tokens)
}

fn parse_add_sub(tokens: &[Token], pos: &mut usize, state: &CalcState) -> Result<f64, String> {
    let mut left = parse_mul_div(tokens, pos, state)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Plus => {
                *pos += 1;
                left += parse_mul_div(tokens, pos, state)?;
            }
            Token::Minus => {
                *pos += 1;
                left -= parse_mul_div(tokens, pos, state)?;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_mul_div(tokens: &[Token], pos: &mut usize, state: &CalcState) -> Result<f64, String> {
    let mut left = parse_power(tokens, pos, state)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Star => {
                *pos += 1;
                left *= parse_power(tokens, pos, state)?;
            }
            Token::Slash => {
                *pos += 1;
                let right = parse_power(tokens, pos, state)?;
                if right == 0.0 {
                    return Err("division by zero".to_string());
                }
                left /= right;
            }
            Token::Percent => {
                *pos += 1;
                let right = parse_power(tokens, pos, state)?;
                if right == 0.0 {
                    return Err("modulo by zero".to_string());
                }
                left %= right;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_power(tokens: &[Token], pos: &mut usize, state: &CalcState) -> Result<f64, String> {
    let base = parse_unary(tokens, pos, state)?;
    if *pos < tokens.len() {
        if let Token::Caret = &tokens[*pos] {
            *pos += 1;
            let exp = parse_unary(tokens, pos, state)?;
            return Ok(base.powf(exp));
        }
    }
    Ok(base)
}

fn parse_unary(tokens: &[Token], pos: &mut usize, state: &CalcState) -> Result<f64, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of expression".to_string());
    }
    match &tokens[*pos] {
        Token::Minus => {
            *pos += 1;
            Ok(-parse_primary(tokens, pos, state)?)
        }
        Token::Plus => {
            *pos += 1;
            parse_primary(tokens, pos, state)
        }
        _ => parse_primary(tokens, pos, state),
    }
}

fn parse_primary(tokens: &[Token], pos: &mut usize, state: &CalcState) -> Result<f64, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of expression".to_string());
    }
    match &tokens[*pos] {
        Token::Number(v) => {
            let v = *v;
            *pos += 1;
            Ok(v)
        }
        Token::LParen => {
            *pos += 1;
            let val = parse_add_sub(tokens, pos, state)?;
            if *pos >= tokens.len() {
                return Err("missing closing parenthesis".to_string());
            }
            if let Token::RParen = &tokens[*pos] {
                *pos += 1;
                Ok(val)
            } else {
                Err("expected closing parenthesis".to_string())
            }
        }
        Token::Ident(name) => {
            let name = name.clone();
            *pos += 1;
            // Check if it's a function call
            if *pos < tokens.len() {
                if let Token::LParen = &tokens[*pos] {
                    *pos += 1; // skip (
                    let arg = parse_add_sub(tokens, pos, state)?;
                    // Check for second arg (min, max)
                    let mut arg2 = None;
                    if *pos < tokens.len() {
                        if let Token::Comma = &tokens[*pos] {
                            *pos += 1;
                            arg2 = Some(parse_add_sub(tokens, pos, state)?);
                        }
                    }
                    if *pos >= tokens.len() {
                        return Err("missing closing parenthesis for function".to_string());
                    }
                    if let Token::RParen = &tokens[*pos] {
                        *pos += 1;
                    } else {
                        return Err("expected closing parenthesis for function".to_string());
                    }
                    return eval_function(&name, arg, arg2, state);
                }
            }
            // Variable lookup
            if let Some(&val) = state.vars.get(&name) {
                Ok(val)
            } else {
                Err(format!("undefined variable: {}", name))
            }
        }
        other => Err(format!("unexpected token: {:?}", other)),
    }
}

fn eval_function(
    name: &str,
    arg: f64,
    arg2: Option<f64>,
    state: &CalcState,
) -> Result<f64, String> {
    let upper = name.to_uppercase();
    match upper.as_str() {
        "SIN" => Ok(state.angle_to_rad(arg).sin()),
        "COS" => Ok(state.angle_to_rad(arg).cos()),
        "TAN" => Ok(state.angle_to_rad(arg).tan()),
        "ASIN" => Ok(state.rad_to_angle(arg.asin())),
        "ACOS" => Ok(state.rad_to_angle(arg.acos())),
        "ATAN" => {
            if let Some(a2) = arg2 {
                Ok(state.rad_to_angle(arg.atan2(a2)))
            } else {
                Ok(state.rad_to_angle(arg.atan()))
            }
        }
        "SINH" => Ok(arg.sinh()),
        "COSH" => Ok(arg.cosh()),
        "TANH" => Ok(arg.tanh()),
        "LOG" | "LOG10" => Ok(arg.log10()),
        "LN" => Ok(arg.ln()),
        "EXP" => Ok(arg.exp()),
        "SQRT" => Ok(arg.sqrt()),
        "ABS" => Ok(arg.abs()),
        "INT" => Ok(arg.trunc()),
        "FLOAT" => Ok(arg),
        "SIGN" => Ok(arg.signum()),
        "MAX" => {
            if let Some(a2) = arg2 {
                Ok(arg.max(a2))
            } else {
                Err("max requires two arguments".to_string())
            }
        }
        "MIN" => {
            if let Some(a2) = arg2 {
                Ok(arg.min(a2))
            } else {
                Err("min requires two arguments".to_string())
            }
        }
        _ => Err(format!("unknown function: {}", name)),
    }
}

fn main() {
    let args = Args::parse();
    let mut state = CalcState::new(args.degrees);

    // If expression given on command line, evaluate and exit
    if let Some(expr) = &args.expression {
        match state.evaluate(expr) {
            Ok(result) => {
                print_result(expr, result);
                process::exit(0);
            }
            Err(e) => {
                eprintln!("ERROR: calc - {}", e);
                process::exit(1);
            }
        }
    }

    // Interactive mode
    let stdin = io::stdin();
    let stdout = io::stdout();
    loop {
        {
            let mut out = stdout.lock();
            let _ = write!(out, " Calc? ");
            let _ = out.flush();
        }
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }
        let line = line.trim();
        if line.is_empty() {
            process::exit(0);
        }
        if line.starts_with('#') || line.starts_with('!') {
            continue;
        }
        let lower = line.to_lowercase();
        if lower == "radians" {
            state.use_degrees = false;
            println!(" OK, We'll use radians");
            continue;
        }
        if lower == "degrees" {
            state.use_degrees = true;
            println!(" OK, We'll use degrees");
            continue;
        }
        if lower == "help" {
            println!(" Possible examples are");
            println!("  SAVEIT=LOG(10)*5");
            println!("  Degrees");
            println!("  X=2*Sin(45)");
            println!("  LN(pi*X)");
            println!(" Etc.");
            println!(" # and ! can be used to start a comment");
            continue;
        }

        match state.evaluate(line) {
            Ok(result) => print_result(line, result),
            Err(e) => eprintln!(" Error: {}", e),
        }
    }
}

fn print_result(expr: &str, result: f64) {
    if result != 0.0 && (result.abs() < 1e-4 || result.abs() > 1e6) {
        println!(" {} = {:16.8E}", expr, result);
    } else if (result.trunc() - result).abs() == 0.0 {
        println!(" {} = {}", expr, result as i64);
    } else {
        println!(" {} = {:.6}", expr, result);
    }
}
