# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:37:52 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
import textwrap
# =============================================================================
# write doc strings
# =============================================================================
def wrap_description(description, column_width):
    """
    split a description into separate lines
    """
    d_lines = textwrap.wrap(description, column_width)
    if len(d_lines) < 9:
        d_lines += [""] * (9 - len(d_lines))

    return d_lines


def write_lines(attr_dict, c1=45, c2=45, c3=15):
    """
     write table lines
    :param lines_list: DESCRIPTION

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

    for key, entry in attr_dict.items():
        d_lines = wrap_description(entry["description"], c2)
        e_lines = wrap_description(entry["example"], c3)
        # line 1 is with the entry
        lines.append(line.format(f"**{key}**", c1, d_lines[0], c2, e_lines[0], c3))
        # line 2 skip an entry in the
        lines.append(line.format("", c1, d_lines[1], c2, e_lines[1], c3))
        # line 3 required
        lines.append(
            line.format(
                f"Required: {entry['required']}", c1, d_lines[2], c2, e_lines[2], c3
            )
        )
        # line 4 blank
        lines.append(line.format("", c1, d_lines[3], c2, e_lines[3], c3))

        # line 5 units
        lines.append(
            line.format(f"Units: {entry['units']}", c1, d_lines[4], c2, e_lines[4], c3)
        )

        # line 6 blank
        lines.append(line.format("", c1, d_lines[5], c2, e_lines[5], c3))

        # line 7 type
        lines.append(
            line.format(f"Type: {entry['type']}", c1, d_lines[6], c2, e_lines[6], c3)
        )

        # line 8 blank
        lines.append(line.format("", c1, d_lines[7], c2, e_lines[7], c3))

        # line 9 type
        lines.append(
            line.format(f"Style: {entry['style']}", c1, d_lines[8], c2, e_lines[8], c3)
        )

        # line 10 blank
        if len(d_lines) > 9:
            lines.append(line.format("", c1, d_lines[9], c2, "", c3))
            for d_line in d_lines[10:]:
                lines.append(line.format("", c1, d_line, c2, "", c3))

        lines.append(hline)

    return "\n".join(lines)
