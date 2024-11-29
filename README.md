# Html syntax Highlight

A simple program for syntax highlighting. Work in progress.

- Tokenize code ✔
- Tag common symbols ✔
- Tag symbols based on context ✔
- format html ✔
- CSS and colors (✔)

## Todo

- Reset indentation: avoid unnecessary indentation of all lines
- Feature based classifier
  - Features
  - Training data utility (...)

## Categories

Some basic classes are captured through deterministic regex operations

- `nu`: Number - decimal, integer, scientific, hex, bin
- `ws`: space, tab
- `nl`: new line
- `pu`: punctuation (,;.:)
- `br`: brackets, in addition these can be counted as layers
- `op`: some basic operators
  - `op-ma`: math operators (+-% etc)

Other classes are captured through a ML model (TODO)

- `kw`: keywords
  - `kw-type`: indicating datatypes?
  - `kw-flow`: code flow (if, for, loop)? probably too specific?
- `lit`: literals: true/false...
- `va`: variables
- `fn`: functions
- `op-?`: less straightforward operators
  - `op-as`: assignment operators (=<- etc)
- `st`: strings
- `co`: comments
- `mo`: modules, importing things?
- `an`: annotations/decorators: @Override in Java or rust derives...
- `uk`: unknown, tokens we cannot predict well.
