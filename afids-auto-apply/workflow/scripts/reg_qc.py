#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import os
import re
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.datasets import load_mni152_template
from svgutils.compose import Unit
from svgutils.transform import GroupElement, SVGFigure, fromstring


def svg2str(display_object, dpi):
    """Serialize a nilearn display object to string."""

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


def output_html(template, input_img, output_html):
    html_list = []
    for ifloat in input_img:
        isub = os.path.basename(ifloat).split("_")[0]

        float_img = nib.load(ifloat)
        float_img = nib.Nifti1Image(
            float_img.get_fdata().astype(np.float32),
            header=float_img.header,
            affine=float_img.affine,
        )

        plot_args_ref = {"dim": 1}

        display = plotting.plot_anat(
            float_img,
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
            "vmin": float_img.get_fdata(dtype="float32").min(),
            "vmax": float_img.get_fdata(dtype="float32").max(),
            "cmap": plt.cm.gray,
            "interpolation": "none",
            "draw_cross": False,
        }

        display = plotting.plot_anat(float_img, **anat_params)
        display.add_contours(template, colors="r", alpha=0.7, linewidths=0.8)

        tmpfile = BytesIO()
        display.savefig(tmpfile, dpi=300)
        display.close()
        tmpfile.seek(0)
        encoded = base64.b64encode(tmpfile.getvalue())

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

    with open(output_html, "w") as fid:
        fid.write(message)


if __name__ == "__main__":
    template = load_mni152_template(resolution=1.00)

    output_html(
        template=template,
        input_img=snakemake.input.images,
        output_html=snakemake.output.html_fig,
    )
