#!/usr/bin/env python

import sys
import re
ENCODING = "ISO-8859-1"

diafname = sys.argv[1]

fl = open(diafname, encoding=ENCODING).readlines()

# Extract lines that describe a function and that have a length information for the function, e.g. lines such as
#   Function       : static, [0218B370][0001:0218A370], len = 0000003B, public: __cdecl std::_Lockit::_Lockit(int)
# and
#   Function: [017C6EC0][0001:017C5EC0] memset(_memset)
of = open(diafname+'.functions_re2.txt', 'w')
for line in fl:
    m = re.match('.*Function.*:[^\[]+\[[0-9A-Fa-f]{8,18}\].*', line)
    if m is not None:
        #print(line)
        of.write(line)
of.close()

