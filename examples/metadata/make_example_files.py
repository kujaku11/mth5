# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:25:38 2020

@author: jpeacock
"""

from xml.dom import minidom
from xml.etree import cElementTree as et
from mth5 import metadata
from pathlib import Path

p = Path(r"c:\Users\jpeacock\Documents\GitHub\mth5_test_data\metadata_files")

style_list = ["json", "xml"]

for style in style_list:
    for c in ["Survey", "Station", "Run", "Channel", "Electric", "Magnetic"]:
        s = getattr(metadata, c)()
        if style == "xml":
            x = s.to_xml(required=False)
            with open(p.joinpath("{0}_example.xml".format(c.lower())), "w") as fid:
                fid.write(
                    minidom.parseString(et.tostring(x).decode()).toprettyxml(
                        indent="    "
                    )
                )
        elif style == "json":
            jstr = s.to_json(nested=True, required=False)
            with open(p.joinpath("{0}_example.json".format(c.lower())), "w") as fid:
                fid.write(jstr)
