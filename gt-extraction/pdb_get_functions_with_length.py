#!/usr/bin/env python

"""
Extracts functions from the output of `dia2dump -all` that contain the starting address and the length of a function.
"""

import sys
import re
ENCODING = "ISO-8859-1"

diafname = sys.argv[1]

fl = open(diafname, encoding=ENCODING).readlines()

# Extract lines that describe a function and that have a length information for the function, e.g. lines such as
#   Function       : static, [0218B370][0001:0218A370], len = 0000003B, public: __cdecl std::_Lockit::_Lockit(int)
of = open(diafname+'.functions_re.txt', 'w')
for line in fl:
    m = re.match('.*Function.*:[^\[]+\[[0-9A-Fa-f]{8,18}\].*len = .*', line)
    if m is not None:
        #print(line)
        of.write(line)
of.close()

