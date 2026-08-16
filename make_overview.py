"""Build a short, plain-language overview PDF of the project."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ACCENT = colors.HexColor("#1F4E79")
MUTED = colors.HexColor("#5A6472")
RULE = colors.HexColor("#D5DAE0")
BOXBG = colors.HexColor("#F2F5F8")

ss = getSampleStyleSheet()
title = ParagraphStyle("t", parent=ss["Title"], fontSize=22, leading=26,
                       textColor=ACCENT, spaceAfter=2)
subtitle = ParagraphStyle("st", parent=ss["Normal"], fontSize=11.5, leading=15,
                          textColor=MUTED, alignment=1, spaceAfter=14)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, leading=17,
                    textColor=ACCENT, spaceBefore=15, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, leading=14,
                    textColor=colors.HexColor("#2E3B48"), spaceBefore=10, spaceAfter=4)
body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10.2, leading=15,
                      alignment=TA_JUSTIFY, spaceAfter=7)
bullet = ParagraphStyle("bu", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=4)
caption = ParagraphStyle("c", parent=ss["Normal"], fontSize=8.8, leading=11.5,
                         textColor=MUTED, alignment=1, spaceBefore=4)
boxstyle = ParagraphStyle("bx", parent=body, fontSize=10, leading=14, spaceAfter=0)


def rule(space_before=2, space_after=8):
    t = Table([[""]], colWidths=[16.4 * cm], rowHeights=[0.6])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.8, RULE)]))
    return [Spacer(1, space_before), t, Spacer(1, space_after)]


def callout(text):
    p = Paragraph(text, boxstyle)
    t = Table([[p]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BOXBG),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def datatable(rows, col_widths, align_right_from=1):
    t = Table(rows, colWidths=col_widths, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (align_right_from, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7F9FB")))
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2.3 * cm, 1.2 * cm, "Ex1 - Visual Navigation for Drones | Intro to Navigation")
    canvas.drawRightString(A4[0] - 2.3 * cm, 1.2 * cm, f"Page {canvas.getPageNumber()}")
    canvas.setStrokeColor(RULE)
    canvas.line(2.3 * cm, 1.6 * cm, A4[0] - 2.3 * cm, 1.6 * cm)
    canvas.restoreState()


story = []

# ---------------------------------------------------------------- page 1
story.append(Paragraph("Finding a Drone Without GPS", title))
story.append(Paragraph("Ex1 &mdash; Visual Navigation for Drones &nbsp;|&nbsp; project overview", subtitle))
story += rule(0, 4)

story.append(Paragraph("1. The problem", h1))
story.append(Paragraph(
    "A drone normally knows where it is because it listens to GPS satellites. That signal is weak "
    "and easy to break: it can be jammed, spoofed, or simply lost between buildings and hills. "
    "The moment it goes, the drone is blind about its own position &mdash; even though its camera "
    "is still working perfectly.", body))
story.append(Paragraph(
    "So the question this project answers is simple to state:", body))
story.append(callout(
    "<b>Given only the video coming out of a drone &mdash; no GPS &mdash; can we work out where it is?</b>"))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "The assignment adds one fair and very reasonable condition: we are allowed to prepare in advance. "
    "Before the GPS-less flight, we get a recording of the same area <i>with</i> GPS. In other words, "
    "we are allowed to build a map first, and then use it.", body))

story.append(Paragraph("2. The idea behind the approach", h1))
story.append(Paragraph(
    "The approach is the same trick a hiker uses with a paper map: you look at the landscape around "
    "you, find that same landscape on the map, and read your position off the map.", body))
story.append(Paragraph("Step one &mdash; build the map (done in advance, on the ground)", h2))
story.append(Paragraph(
    "Take the earlier flight, where GPS was available. Cut its video into still frames, one per second. "
    "For each frame we already know exactly where the drone was, because the flight log tells us. "
    "Then, for every frame, the computer picks out a few thousand tiny distinctive spots &mdash; a corner "
    "of a roof, the end of a road marking, the edge of a tree. These are called <i>features</i>. "
    "The result is a small library of pictures, each one tagged with a real-world coordinate.", body))
story.append(Paragraph("Step two &mdash; navigate (done in the air, live, with no GPS)", h2))
story.append(Paragraph(
    "A new frame arrives from the camera. The computer finds the same kind of distinctive spots in it, "
    "and asks: which picture in my library shows the same place? Once it finds the match, it works out "
    "exactly how the new picture is shifted relative to the library picture, converts that shift from "
    "pixels into metres, and adds it to the library picture's known coordinate. That gives the drone's "
    "position.", body))
story.append(Paragraph(
    "Two extra ingredients stop this from going badly wrong in practice:", body))
story.append(Paragraph(
    "<b>A sense of momentum.</b> A drone cannot teleport. The system keeps track of roughly where the "
    "drone should be, based on where it was a second ago and how fast it was moving, and only searches "
    "the part of the map that is physically within reach. This kills the classic failure where a car "
    "park matches a different, identical-looking car park a kilometre away.", bullet, bulletText="•"))
story.append(Paragraph(
    "<b>Knowing when not to trust itself.</b> A match is only accepted if enough of the matched spots "
    "agree with each other geometrically. If they do not, the system says so and coasts on momentum "
    "for a moment rather than reporting a confident, wrong answer.", bullet, bulletText="•"))

story.append(PageBreak())

# ---------------------------------------------------------------- page 2
story.append(Paragraph("3. How it was solved &mdash; and what went wrong first", h1))
story.append(Paragraph(
    "A first working version was built exactly as described above. It ran end to end, but its accuracy "
    "was poor: it was typically wrong by about 40 metres, and sometimes by 200. Investigating <i>why</i> "
    "turned out to be the most interesting part of the project, because the two causes were not bugs in "
    "the code &mdash; they were wrong assumptions.", body))

story.append(Paragraph("Problem A: the ruler was wrong", h2))
story.append(Paragraph(
    "To turn a shift measured in pixels into a distance in metres, you need to know how many metres one "
    "pixel covers. The first version calculated this from the camera's field of view and the altitude "
    "written in the flight log. But that altitude is measured from the <i>take-off point</i>, not from the "
    "ground the camera is actually looking at &mdash; and the flight log for these drones does not record "
    "the camera's tilt angle at all. The ruler was about 36% too short, so every distance came out too small.", body))
story.append(Paragraph(
    "<b>The fix:</b> stop guessing and measure it. During the preparation stage we already know the real "
    "GPS positions, so we can compare how far the drone truly moved between two frames with how far the "
    "picture shifted, and read the true metres-per-pixel straight off the data. As a sanity check, the "
    "same number was confirmed independently from the image itself, by measuring the width of parking "
    "bays in the footage.", body))

story.append(Paragraph("Problem B: the test was measuring the wrong thing", h2))
story.append(Paragraph(
    "The original experiment used the first 80% of a flight as the map and the last 20% as the test. "
    "But on these flights the last stretch is the drone flying <i>back home over ground it has already "
    "covered, facing the opposite direction</i>. A roof looks completely different when you approach it "
    "from the other side &mdash; different shadows, different faces of buildings visible &mdash; and the "
    "matching method used here simply cannot recognise it.", body))
story.append(Paragraph(
    "This was confirmed by cheating deliberately: even when the system was <i>told</i> the correct answer "
    "and handed the right map picture, it still could not match it. So the 40-metre figure was never "
    "measuring how well the navigator works. It was measuring a limitation of the matching method.", body))
story.append(Paragraph(
    "<b>The fix:</b> test it properly. Hold out every fifth frame as the test set, keep the rest as the "
    "map, and forbid the system from matching a test frame against any map frame recorded within two "
    "seconds of it, so it cannot cheat by matching its own immediate neighbour.", body))

story.append(Paragraph("4. The result", h1))
story.append(Paragraph(
    "With the ruler corrected and the experiment set up honestly, the system locates the drone from its "
    "camera alone to within a few metres &mdash; and it produces a confident visual fix on essentially "
    "every single frame, instead of frequently giving up:", body))
story.append(Spacer(1, 4))
story.append(datatable([
    ["Flight", "Typical error\nBEFORE", "Typical error\nAFTER", "Frames located\nvisually"],
    ["flight (50 m altitude)", "40 m", "6 m", "23 of 23"],
    ["flight_0024 (31 m altitude)", "17 m", "10 m", "26 of 27"],
], [6.0 * cm, 3.5 * cm, 3.5 * cm, 3.4 * cm]))
story.append(Paragraph(
    "\"Typical error\" is the median distance between the estimated position and the true GPS position.",
    caption))

story.append(PageBreak())

# ---------------------------------------------------------------- page 3
story.append(Paragraph("5. Seeing it work", h1))
story.append(Paragraph(
    "The blue line is the route the drone actually flew, according to its GPS log. The blue dots are the "
    "moments we tested. The red crosses are where the system <i>thought</i> the drone was, using nothing "
    "but the camera. The closer each cross sits to its dot, the better.", body))
story.append(Spacer(1, 6))
story.append(Image("results/flight/trajectory_comparison.png", width=9.9 * cm, height=9.9 * cm))
story.append(Paragraph(
    "Flight over a university campus, roughly 350 m across, flown at about 50 m altitude.", caption))

story.append(Spacer(1, 2))
story.append(Paragraph("6. What it still cannot do", h1))
story.append(Paragraph(
    "Honest limitations are worth stating, because they point at the next piece of work:", body))
story.append(Paragraph(
    "<b>Coming back the other way.</b> The system still fails to recognise a place it is revisiting from "
    "the opposite direction. This is a known weakness of the classical matching method used here. The "
    "standard modern remedy is to swap it for a learned, AI-based matcher, which is far better at "
    "recognising a place from an unfamiliar angle. The code is arranged so this is a change to one "
    "function.", bullet, bulletText="•"))
story.append(Paragraph(
    "<b>One ruler per flight.</b> The metres-per-pixel figure is a single number for the whole flight. "
    "When the camera is tilted, the far side of the picture is genuinely further away than the near "
    "side, so the estimate is least accurate towards the edges of the frame.", bullet, bulletText="•"))
story.append(Paragraph(
    "<b>A map is still required.</b> This is navigation <i>within a known area</i>. The drone cannot be "
    "dropped somewhere it has never seen. Extending the map from \"an earlier flight\" to satellite "
    "imagery such as Google Earth is the natural next step, and is what the follow-on project targets.",
    bullet, bulletText="•"))

story.append(Spacer(1, 6))
story += rule(0, 4)
story.append(Paragraph(
    "<b>Checking the work.</b> <font face='Courier'>python summarize.py</font> re-prints the numbers in "
    "this document straight from the stored result files, with nothing to install. To re-run the whole "
    "pipeline from scratch, put the flight videos in <font face='Courier'>data/raw/</font> and run "
    "<font face='Courier'>./reproduce.sh</font>. <font face='Courier'>RESULTS.md</font> holds the full "
    "technical write-up, including the evidence for each of the two problems described above.", body))

doc = SimpleDocTemplate(
    "PROJECT_OVERVIEW.pdf", pagesize=A4,
    leftMargin=2.3 * cm, rightMargin=2.3 * cm,
    topMargin=1.9 * cm, bottomMargin=2.1 * cm,
    title="Finding a Drone Without GPS - Project Overview",
    author="Ex1 - Visual Navigation for Drones",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("written PROJECT_OVERVIEW.pdf")
