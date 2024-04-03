text = self.pre_process(text)
variables, p_variable = self.find_variables(text)
if variables:
    self.patterns_find["variable"] = p_variable

text = self.syntax_highlight(text)