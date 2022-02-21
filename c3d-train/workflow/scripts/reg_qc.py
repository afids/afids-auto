#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import image, plotting
from nilearn.datasets import load_mni152_template

matplotlib.use("Agg")
import base64
import re
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from svgutils.compose import Unit
from svgutils.transform import GroupElement, SVGFigure, fromstring

template = load_mni152_template(resolution=1.00)


def svg2str(display_object, dpi):
    """Serialize a nilearn display object to string."""
    from io import StringIO

    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(
        image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k"
    )
    return image_buf.getvalue()


def extract_svg(display_object, dpi=250):
    """Remove the preamble of the svg files generated with nilearn."""
    image_svg = svg2str(display_object, dpi)

    image_svg = re.sub(' height="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(' width="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(
        " viewBox", ' preseveAspectRation="xMidYMid meet" viewBox', image_svg, count=1
    )
    start_tag = "<svg "
    start_idx = image_svg.find(start_tag)
    end_tag = "</svg>"
    end_idx = image_svg.rfind(end_tag)

    # rfind gives the start index of the substr. We want this substr
    # included in our return value so we add its length to the index.
    end_idx += len(end_tag)
    return image_svg[start_idx:end_idx]


def clean_svg(fg_svgs, bg_svgs, ref=0):
    # Find and replace the figure_1 id.
    svgs = bg_svgs + fg_svgs
    roots = [f.getroot() for f in svgs]

    sizes = []
    for f in svgs:
        viewbox = [float(v) for v in f.root.get("viewBox").split(" ")]
        width = int(viewbox[2])
        height = int(viewbox[3])
        sizes.append((width, height))
    nsvgs = len([bg_svgs])

    sizes = np.array(sizes)

    # Calculate the scale to fit all widths
    width = sizes[ref, 0]
    scales = width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    fig = SVGFigure(Unit(f"{width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

    yoffset = 0
    for i, r in enumerate(roots):
        r.moveto(0, yoffset, scale_x=scales[i])
        if i == (nsvgs - 1):
            yoffset = 0
        else:
            yoffset += heights[i]

    # Group background and foreground panels in two groups
    if fg_svgs:
        newroots = [
            GroupElement(roots[:nsvgs], {"class": "background-svg"}),
            GroupElement(roots[nsvgs:], {"class": "foreground-svg"}),
        ]
    else:
        newroots = roots

    fig.append(newroots)
    fig.root.attrib.pop("width", None)
    fig.root.attrib.pop("height", None)
    fig.root.set("preserveAspectRatio", "xMidYMid meet")

    with TemporaryDirectory() as tmpdirname:
        out_file = Path(tmpdirname) / "tmp.svg"
        fig.save(str(out_file))
        # Post processing
        svg = out_file.read_text().splitlines()

    # Remove <?xml... line
    if svg[0].startswith("<?xml"):
        svg = svg[1:]

    # Add styles for the flicker animation
    if fg_svgs:
        svg.insert(
            2,
            """\
<style type="text/css">
@keyframes flickerAnimation%s { 0%% {opacity: 1;} 100%% { opacity:0; }}
.foreground-svg { animation: 1s ease-in-out 0s alternate none infinite running flickerAnimation%s;}
.foreground-svg:hover { animation-play-state: running;}
</style>"""
            % tuple([uuid4()] * 2),
        )

    return svg


def sorted_nicely(data, reverse=False):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key, reverse=reverse)


#%%


dataset = "clinical"
dir_data = (
    f"/home/greydon/Documents/data/autoafids/{dataset}/derivatives/run_2022-02-22"
)

html_list = []
for isub in sorted_nicely(
    [
        x
        for x in os.listdir(os.path.join(dir_data, "reg_aladin"))
        if os.path.isdir(os.path.join(dir_data, "reg_aladin", x))
    ]
):
    ref = glob.glob(os.path.join(dir_data, "reg_aladin", isub, "anat") + "/*.nii.gz")

    if len(ref) == 1:
        ref_img = nib.load(ref[0])
        ref_img = nib.Nifti1Image(
            ref_img.get_fdata().astype(np.float32),
            header=ref_img.header,
            affine=ref_img.affine,
        )

        plot_args_ref = {"dim": 1}

        display = plotting.plot_anat(
            ref_img,
            display_mode="ortho",
            draw_cross=False,
            cut_coords=[0, 0, 40],
            **plot_args_ref,
        )
        fg_svgs = [fromstring(extract_svg(display, 300))]
        display.close()

        display = plotting.plot_anat(
            template,
            display_mode="ortho",
            draw_cross=False,
            cut_coords=[0, 0, 40],
            **plot_args_ref,
        )
        bg_svgs = [fromstring(extract_svg(display, 300))]
        display.close()

        final_svg = "\n".join(clean_svg(fg_svgs, bg_svgs))

        anat_params = {
            "vmin": ref_img.get_fdata(dtype="float32").min(),
            "vmax": ref_img.get_fdata(dtype="float32").max(),
            "cmap": plt.cm.gray,
            "interpolation": "none",
            "draw_cross": False,
        }

        display = plotting.plot_anat(ref_img, **anat_params)
        display.add_contours(template, colors="r", alpha=0.7, linewidths=0.8)

        tmpfile = BytesIO()
        display.savefig(tmpfile, dpi=300)
        display.close()
        tmpfile.seek(0)
        encoded = base64.b64encode(tmpfile.getvalue())

        tmpfile_ref = BytesIO()
        display.savefig(tmpfile_ref, dpi=300)
        display.close()
        tmpfile_ref.seek(0)
        data_uri = base64.b64encode(tmpfile_ref.getvalue()).decode("utf-8")
        img_tag = '<center><img src="data:image/png;base64,{0}"/></center>'.format(
            data_uri
        )

        html_list.append(
            f"""
				<center>
					<h1 style="font-size:42px">{isub}</h1>
					<p>{final_svg}</p>
					<p><img src="data:image/png;base64, {encoded.decode("utf-8")}" width=1600 height=600></p>
					<hr style="height:4px;border-width:0;color:black;background-color:black;margin:30px;">
				</center>"""
        )

        print(f"Done {isub}")

html_string = "".join(html_list)
message = f"""<html>
		<head></head>
		<body>{html_string}</body>
		</html>"""

with open(
    f"/home/greydon/Downloads/{dataset}_from-subject_to-MNI152NLin2009cAsym_regqc.html",
    "w",
) as fid:
    fid.write(message)
