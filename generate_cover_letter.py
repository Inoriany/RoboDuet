from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = "Yuxiao_Zhao_CoverLetter_Apple.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2.5*cm,
    rightMargin=2.5*cm,
    topMargin=2.2*cm,
    bottomMargin=2.2*cm,
)

styles = getSampleStyleSheet()

# --- Custom styles ---
name_style = ParagraphStyle(
    "Name",
    fontSize=18,
    fontName="Helvetica-Bold",
    textColor=colors.HexColor("#1a1a1a"),
    spaceBefore=10,
    spaceAfter=8,
    alignment=TA_LEFT,
)

contact_style = ParagraphStyle(
    "Contact",
    fontSize=9,
    fontName="Helvetica",
    textColor=colors.HexColor("#555555"),
    spaceAfter=4,
    alignment=TA_LEFT,
)

body_style = ParagraphStyle(
    "Body",
    fontSize=10,
    fontName="Helvetica",
    textColor=colors.HexColor("#1a1a1a"),
    leading=15,
    spaceAfter=10,
    alignment=TA_JUSTIFY,
)

bold_body_style = ParagraphStyle(
    "BoldBody",
    parent=body_style,
    fontName="Helvetica-Bold",
)

section_label_style = ParagraphStyle(
    "SectionLabel",
    fontSize=9,
    fontName="Helvetica-Bold",
    textColor=colors.HexColor("#888888"),
    spaceAfter=6,
    spaceBefore=12,
    alignment=TA_LEFT,
)

date_style = ParagraphStyle(
    "Date",
    fontSize=10,
    fontName="Helvetica",
    textColor=colors.HexColor("#555555"),
    spaceAfter=14,
    alignment=TA_LEFT,
)

salutation_style = ParagraphStyle(
    "Salutation",
    fontSize=10,
    fontName="Helvetica-Bold",
    textColor=colors.HexColor("#1a1a1a"),
    spaceAfter=10,
    alignment=TA_LEFT,
)

sign_style = ParagraphStyle(
    "Sign",
    fontSize=10,
    fontName="Helvetica",
    textColor=colors.HexColor("#1a1a1a"),
    leading=15,
    spaceAfter=3,
    alignment=TA_LEFT,
)

sign_name_style = ParagraphStyle(
    "SignName",
    fontSize=10,
    fontName="Helvetica-Bold",
    textColor=colors.HexColor("#1a1a1a"),
    spaceAfter=2,
    alignment=TA_LEFT,
)

# --- Content ---
story = []

# Header
story.append(Paragraph("Yuxiao Zhao", name_style))
story.append(Paragraph("zwb8@163.com &nbsp;&nbsp;|&nbsp;&nbsp; +852 44898874 (Hong Kong) &nbsp;&nbsp;|&nbsp;&nbsp; +86 13316988235 (China)", contact_style))
story.append(Paragraph("MSc in Artificial Intelligence, The Chinese University of Hong Kong", contact_style))
story.append(Spacer(1, 0.5*cm))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceAfter=14))

# Date
story.append(Paragraph("March 2026", date_style))

# Salutation
story.append(Paragraph("Dear Apple Hiring Team,", salutation_style))

# Body paragraphs
paragraphs = [
    (False, "I am writing to apply for the <b>Computer Vision/Machine Learning Intern (Agentic AI)</b> position within Apple's Video Engineering organization. I am currently an MSc student in Artificial Intelligence at The Chinese University of Hong Kong, where I conduct research under Prof. Chen Fei at the CLOVER Lab. My work spans agentic AI, multi-modal perception, and embodied control — directly aligned with the focus areas of this role."),

    (False, "My flagship research project, <b>VLA-LocoMAN</b>, is an end-to-end controller learning system for quadruped locomotion-manipulation tasks. At its core, I designed a vision-language-action (VLA) interface that grounds natural language instructions in visual context and maps them to continuous robot control actions — a direct instantiation of the Agentic AI and Multi-Modal LLM paradigm that Apple's team is pursuing. The system is built on a hierarchical reinforcement learning framework (PPO/SAC) with curriculum learning and reward shaping, and includes a sim-to-real transfer pipeline for deployment on physical robots."),

    (False, "Complementing this, I am currently working on a <b>Text-to-Image Generation with RL</b> project, which uses multimodal LLMs as reward models — including VLM-based object/attribute correctness and CLIP similarity — to align generation quality via GRPO-style reward-guided training. This project directly engages with Video Foundation Models, generative editing, and multi-modal reward modeling — core themes of Apple's Video Engineering work. It has deepened my practical understanding of where alignment methods succeed and fail beyond leaderboard scores."),

    (False, "My broader LLM work includes hands-on experience with Transformer architectures, diffusion models, LoRA fine-tuning, instruction tuning, and alignment strategies including RLHF, DPO, and Chain-of-Thought reasoning. I have applied Qwen2.5-0.5B to supervised fine-tuning for downstream tasks and explored world models for environment prediction in reinforcement learning contexts."),

    (False, "My applied computer vision depth is demonstrated through my <b>autonomous fire detection robot</b>, where I engineered a real-time perception stack integrating YOLOv5, RGB/thermal fusion, LiDAR, and SLAM under ROS Noetic, achieving a 93% fire-detection F1 score and 92% closed-loop accuracy. This project challenged me to develop algorithms that perform reliably under latency and resource constraints — a challenge fundamental to Apple's on-device vision platform."),

    (False, "On the research side, I independently proposed the <b>Adaptive Federated Normalization (AFN)</b> algorithm addressing feature distribution shift in federated learning, achieving 98.76% accuracy and outperforming FedAvg and FedBN baselines. This work has been released as a first-author preprint (2025)."),

    (False, "I am proficient in Python, experienced with PyTorch, HuggingFace Transformers, CUDA/Triton, and Docker, and comfortable prototyping across the full stack from model training to deployment. I hold a BSc in Computer Science and Mathematics from The University of Manchester, where I was awarded the <b>Stellify Award</b> (top 5% of students) and achieved runner-up at GreatUniHack 2023."),

    (False, "Apple's position — closing the loop between cutting-edge research and product impact at scale, on-device — is exactly the environment I want to work in. I am confident I can contribute meaningfully to the Video Engineering team."),

    (False, "Thank you for your time and consideration. I would welcome the opportunity to discuss my background further."),
]

for _, text in paragraphs:
    story.append(Paragraph(text, body_style))

# Closing
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Sincerely,", sign_style))
story.append(Spacer(1, 0.6*cm))
story.append(Paragraph("Yuxiao Zhao", sign_name_style))
story.append(Paragraph("zwb8@163.com &nbsp;&nbsp;|&nbsp;&nbsp; +852 44898874 (HK) &nbsp;&nbsp;|&nbsp;&nbsp; +86 13316988235 (CN)", sign_style))

doc.build(story)
print(f"PDF generated: {OUTPUT}")
