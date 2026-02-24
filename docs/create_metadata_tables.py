# -*- coding: utf-8 -*-
"""

Created on Thu Jul 30 17:01:34 2020

:author: Jared Peacock

:license: MIT

"""

import textwrap
from pathlib import Path


fn = Path(r"c:\Users\peaco\Documents\GitHub\mth5\mth5\docs\mt_metadata_guide.tex")


def wrap_description(description, column_width):
    """
    split a description into separate lines
    """
    d_lines = textwrap.wrap(description, column_width)
    if len(d_lines) < 9:
        d_lines += [""] * (9 - len(d_lines))

    return d_lines


def write_lines(lines_list, c1, c2, c3):
    """
     write table lines
    :param lines_list: DESCRIPTION
    :type lines_list: TYPE
    :param c1: DESCRIPTION
    :type c1: TYPE
    :param c2: DESCRIPTION
    :type c2: TYPE
    :param c3: DESCRIPTION
    :type c3: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    line = "       | {0:<{1}}| {2:<{3}} | {4:<{5}}|"
    hline = "       +{0}+{1}+{2}+".format(
        "-" * (c1 + 1), "-" * (c2 + 2), "-" * (c3 + 1)
    )
    mline = "       +{0}+{1}+{2}+".format(
        "=" * (c1 + 1), "=" * (c2 + 2), "=" * (c3 + 1)
    )

    lines = [
        hline,
        line.format("**Metadata Key**", c1, "**Description**", c2, "**Example**", c3),
        mline,
    ]

    for entry in lines_list:
        d_lines = wrap_description(entry[5], c2)
        e_lines = wrap_description(entry[-1], c3)
        # line 1 is with the entry
        lines.append(line.format(f"**{entry[0]}**", c1, d_lines[0], c2, e_lines[0], c3))
        # line 2 skip an entry in the
        lines.append(line.format("", c1, d_lines[1], c2, e_lines[1], c3))
        # line 3 required
        lines.append(
            line.format(f"Required: {entry[1]}", c1, d_lines[2], c2, e_lines[2], c3)
        )
        # line 4 blank
        lines.append(line.format("", c1, d_lines[3], c2, e_lines[3], c3))

        # line 5 units
        lines.append(
            line.format(f"Units: {entry[2]}", c1, d_lines[4], c2, e_lines[4], c3)
        )

        # line 6 blank
        lines.append(line.format("", c1, d_lines[5], c2, e_lines[5], c3))

        # line 7 type
        lines.append(
            line.format(f"Type: {entry[3]}", c1, d_lines[6], c2, e_lines[6], c3)
        )

        # line 8 blank
        lines.append(line.format("", c1, d_lines[7], c2, e_lines[7], c3))

        # line 9 type
        lines.append(
            line.format(f"Style: {entry[4]}", c1, d_lines[8], c2, e_lines[8], c3)
        )

        # line 10 blank
        if len(d_lines) > 9:
            lines.append(line.format("", c1, d_lines[9], c2, "", c3))
            for d_line in d_lines[10:]:
                lines.append(line.format("", c1, d_line, c2, "", c3))

        lines.append(hline)

    return lines


def write_block(entry, c1, c2, c3):
    """
     write table lines
    :param lines_list: DESCRIPTION
    :type lines_list: TYPE
    :param c1: DESCRIPTION
    :type c1: TYPE
    :param c2: DESCRIPTION
    :type c2: TYPE
    :param c3: DESCRIPTION
    :type c3: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    line = "       | {0:<{1}}| {2:<{3}} | {4:<{5}}|"
    hline = "       +{0}+{1}+{2}+".format(
        "-" * (c1 + 1), "-" * (c2 + 2), "-" * (c3 + 1)
    )
    mline = "       +{0}+{1}+{2}+".format(
        "=" * (c1 + 1), "=" * (c2 + 2), "=" * (c3 + 1)
    )
    section = f":navy:`{entry[0]}`"

    lines = [
        section,
        "~" * len(section),
        "",
        ".. container::",
        "",
        "   .. table::",
        "       :class: tight-table",
        f"       :widths: {c1} {c2} {c3}",
        "",
        hline,
        line.format(f"**{entry[0]}**", c1, "**Description**", c2, "**Example**", c3),
        mline,
    ]

    d_lines = wrap_description(entry[5], c2)
    e_lines = wrap_description(entry[-1], c3)
    # line 1 is with the entry
    lines.append(
        line.format(f"**Required**: {entry[1]}", c1, d_lines[0], c2, e_lines[0], c3)
    )
    # line 2 skip an entry in the
    lines.append(line.format("", c1, d_lines[1], c2, e_lines[1], c3))
    # line 3 required
    lines.append(
        line.format(f"**Units**: {entry[2]}", c1, d_lines[2], c2, e_lines[2], c3)
    )
    # line 4 blank
    lines.append(line.format("", c1, d_lines[3], c2, e_lines[3], c3))

    # line 5 units
    lines.append(
        line.format(f"**Type**: {entry[3]}", c1, d_lines[4], c2, e_lines[4], c3)
    )

    # line 6 blank
    lines.append(line.format("", c1, d_lines[5], c2, e_lines[5], c3))

    # line 7 type
    lines.append(
        line.format(f"**Style**: {entry[4]}", c1, d_lines[6], c2, e_lines[6], c3)
    )

    # line 8 blank
    lines.append(line.format("", c1, d_lines[7], c2, e_lines[7], c3))

    # line 9 type
    lines.append(line.format("", c1, d_lines[8], c2, e_lines[8], c3))

    # line 10 blank
    if len(d_lines) > 9:
        lines.append(line.format("", c1, d_lines[9], c2, "", c3))
        for d_line in d_lines[10:]:
            lines.append(line.format("", c1, d_line, c2, "", c3))

    lines.append(hline)
    lines.append("")

    return lines


lines = fn.read_text().split("\n")

entries = {}
for line in lines[170:]:
    if "\section{" in line:
        key = line.strip().split("{")[1].replace("}", "").lower()
        entries[key] = []
    if "\entry" in line:
        entries[key].append(
            line.strip()
            .replace("\\entry{", "")
            .replace("\_", "_")
            .replace("}{", ",")
            .replace("\\True", ":red:`True`")
            .replace("\\False", ":blue:`False`")
            .replace("}", "")
            .split(",")
        )

c1 = 30
c2 = 45
c3 = 15

# for key, value in entries.items():
#     lines = write_lines(value, c1, c2, c3)
#     with open(
#         r"c:\Users\peaco\Documents\{0}_metadata_table.rst".format(key), "w"
#     ) as fid:
#         fid.write("\n".join(lines))

for key, sections in entries.items():
    section_lines = []
    for entry in sections:
        section_lines += write_block(entry, c1, c2, c3)

    with open(
        r"c:\Users\peaco\Documents\{0}_metadata_table.rst".format(key), "w"
    ) as fid:
        fid.write("\n".join(section_lines))
