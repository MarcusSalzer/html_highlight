# Html syntax Highlight

## Todo

- Reset indentation: avoid unnecessary indentation of all lines
- Feature based classifier
  - Features
- inline mode: try to catch code fragments in text
- language classifier: if confident, we can put a little language title at the top of the generated block, especially for longer snippets
- highlighting inside strings

## Classes

Some basic classes are captured through a deterministic (**det**) process. Others are inferred by a model, if possible.

### Keywords

- `kwfl`: flow keyword. if, for, return, try, except
- `kwty`: type keyword. int, f64, void
- `kwop`: operator keyword. Used like operator. in, is, select
- `kwmo`: modifier keyword. pub, private, static, final, volatile
- `kwva`: variable declaration. let, const,
- `kwde`: other declaration. struct, enum, class, module etc
- `kwfn`: function declaration. def, fn, function
- `kwim`: import keyword. import, from, #include, use
- `kwio`: io keyword. echo, print, DISPLAY (quite rare)

### Syntax features

- `id`: indentation (**det**). space/tab at beginning of line
- `ws`: whitespace (**det**). space, tab
- `nl`: new-line (**det**).
- `br`: brackets. In pairs. (**det**) (), [], {}. Sometimes <>.
- `sy`: syntax features (**det**). :, ::, ->, =>, >>>
- `pu`: punctuation (**det**). , ;
- `cofl`: full-line comment.
- `coil`: in-line comment, to end of line.
- `coml`: possibly multi-line comment, with start and end tag.

### Literal values

- `nu`: number (**det**). dec, int, scientific, hex, bin.
- `st`: string. Basic strings (**det**).
- `bo`: boolean.: true, false, True
- `li`: other literal. null, None, undefined

### Operators

- `opcm`: comparison operator.
- `opbi`: binary operator. Other binary operators
- `opun`: unary operator. &ref, !not, X', x++, --x
- `opas`: assignment operators. =, <-, +=,

### Objects and functions

- `pa`: parameter. a variable defined together with a function.
- `cl`: class. Non-primitve defined, also traits.
- `clco`: class constructor. class name used as a function
- `mo`: module/namespace.
- `fnme`: method. A function on an object instance
- `fnas`: associated/static method/function. On module or class
- `fnsa`: standalone function.
- `an`: annotation. @Override, #[ allow() ], @property
- `va`: variable.
- `at`: attribute. a variable/constant on some object or module.

### Other

- `uk`: unknown. To be inferred, or cannot be inferred.

## Example difficulty

Examples are rated as one of three levels.

### Easy

Every token has a well defined tag. A deterministic model can solve them.

### Normal

Every token has a well defined tag. Requires a context aware model to solve them with a high probability.

### Ambiguous

At least one token could have one of multiple tags.
