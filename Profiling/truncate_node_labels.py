import re, pydot

angle_re = re.compile(r'<[^<>]*>')
paren_re = re.compile(r'\([^()]*\)')
ns_re = re.compile(r'\S+::')

def simplify(s):
    parts = s.split('\\n')
    s = parts[1]
    
    # first remove all template < ... > strings
    slast = None
    while slast != s:
        slast = s
        s = angle_re.sub('', s)

    # now remove all function args ( ... ) strings
    slast = None
    while slast != s:
        slast = s
        s = paren_re.sub('', s)

    # now remove all namespace x::y strings 
    slast = None
    while slast != s:
        slast = s
        s = ns_re.sub('', s)
    
    parts[1] = s
    return '\\n'.join(parts)

dotty = pydot.graph_from_dot_file('./foo.dot')

for n in dotty.get_nodes():
    if n.get_label(): 
        #n.set_label(re.sub(',','_', n.get_label()))
        n.set_label(simplify(n.get_label()))

dotty.write_dot('foo2.dot')
