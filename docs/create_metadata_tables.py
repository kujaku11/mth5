# -*- coding: utf-8 -*-
"""

Created on Thu Jul 30 17:01:34 2020

:author: Jared Peacock

:license: MIT

"""

from pathlib import Path
import textwrap

fn = Path(r"c:\Users\peaco\Documents\GitHub\mth5\mth5\docs\mt_metadata_guide.tex")

def wrap_description(description, column_width):
    """
    split a description into separate lines
    """
    d_lines = textwrap.wrap(description, column_width)
    if len(d_lines) < 5:
        d_lines += [''] * (5 - len(d_lines))
        
    return d_lines

lines = fn.read_text().split('\n')

entries = []
for line in lines:
    if '\entry' in line:
        entries.append(line.strip().replace('\\entry{', '').replace('\_', '_').replace('}{', ',').replace('\\True', 'True').replace('\\False', 'False').replace('}', '').split(','))

c1 = max([len(entry[0]) for entry in entries]) + 5
c3 = 30
c2 = 120 - (c1 + c3)
line = '| {0:<{1}}| {2:<{3}} | {4:<{5}}|'
hline = '+{0}+{1}+{2}+'.format('-' * (c1 + 1),
                               '-' * (c2 + 2),
                               '-' * (c3 + 1))
mline = '+{0}+{1}+{2}+'.format('=' * (c1 + 1),
                               '=' * (c2 + 2),
                               '=' * (c3 + 1))

lines = [hline,
         line.format('**Metadata Key**', c1,
                     '**Description**', c2,
                     '**Example**', c3), 
         mline]

for entry in entries[1:]:
    d_lines = wrap_description(entry[5], c2)
    e_lines = wrap_description(entry[-1], c3)
    # line 1 is with the entry
    lines.append(line.format(f'**{entry[0]}**', c1, d_lines[0], c2, 
                             e_lines[0], c3))
    # line 2 skip an entry in the
    lines.append(line.format('', c1, '', c2, '', c3))
    # line 3 required
    lines.append(line.format(f'Required: {entry[1]}', c1,
                             d_lines[1], c2,
                             e_lines[1], c3))
    # line 4 blank
    lines.append(line.format('', c1, '', c2, '', c3))
    
    # line 5 units
    lines.append(line.format(f'Units: {entry[2]}', c1,
                             d_lines[2], c2,
                             e_lines[2], c3))

        
    # line 6 blank
    lines.append(line.format('', c1, '', c2, '', c3))
    
    # line 7 type
    lines.append(line.format(f'Type: {entry[3]}', c1, 
                             d_lines[3], c2,
                             e_lines[3], c3))
    
    # line 8 blank
    lines.append(line.format('', c1, '', c2, '', c3))
    
    # line 9 type
    lines.append(line.format(f'Style: {entry[4]}', c1,
                             d_lines[4], c2,
                             e_lines[4], c3))

    # line 10 blank
    if len(d_lines) > 5:
        lines.append(line.format('', c1, '', c2, '', c3))
        for d_line in d_lines[4:]:
            lines.append(line.format('', c1, d_line, c2,
                                     '', c3))
            lines.append(line.format('', c1, '', c2, '', c3))
        
    lines.append(hline)
        
with open(r"c:\Users\peaco\Documents\metadata_table.rst", 'w') as fid:
    fid.write('\n'.join(lines))    