# Html syntax Highlight

The idea is to use a regex pattern for tokenization and deterministic tagging. Then, a classifier (LSTM etc) can fill in the tags on ambiguous tokens

## Classes

We are trying

### Keywords

- `kwfl`: flow keyword. if, for, return, try, except
- `kwop`: operator keyword. Used like operator. in, is, select, new, echo
- `kwmo`: modifier keyword. pub, private, static, final, volatile
- `kwde`: declare variable, class, function
- `kwim`: import keyword. import, from, #include (?), use

### Syntax features

- `id`: indentation. space/tab at beginning of line
- `ws`: whitespace. space, tab
- `nl`: new-line.
- `brop`: opening brackets
- `brcl`: closing brackets
- `sy`: syntax features. :, ::, ->, =>, >>>, also <> in types
- `pu`: punctuation.
- `co`: comments (inline/multiline/single line)

### Literal values

- `nu`: number. dec, int, scientific, hex, bin, percent.
- `st`: string.
- `bo`: boolean literals.
- `li`: other literal. null, None, undefined, built in constant values

### Operators

- `opbi`: binary operator. Other binary operators
- `opun`: unary operator. &ref, !not, X', x++, --x
- `opas`: assignment operators. =, <-, +=,
- `opmo`: modifier operators. references, pointers etc

### Objects and functions

- `pa`: parameter. a variable defined together with a function.
- `ty`: type keyword. int, f64, void
- `tyco`: type keyword cosntructor.
- `cl`: class. Non-primitve defined, also traits.
- `clco`: class constructor. class name used as a function
- `mo`: module/namespace.
- `fnme`: method. A function on an object instance
- `fnas`: associated/static method/function. On module or class
- `fnfr`: standalone function.
- `fnto`: function tear-off.
- `an`: annotation. @Override, #[ allow() ], @property, rust lifetimes
- `va`: variable or similar user defined identifier.
- `at`: attribute. a variable/constant on some object or module.

### Other

- `uk`: unknown.

## Roadmap

- ✅ LSTM Tagger _24-12-07_
- ✅ Render HTML preview _25-01-19_
- ✅ NDJSON dataset _25-08-30_
- ✅ Cleanup labels, linting _25-09-03_
- Reset indentation: avoid unnecessary indentation of all lines
- RNN variant comparison
- Feature based classifier
- inline mode: try to catch code fragments in text?
- language classifier?
- highlighting inside strings?
