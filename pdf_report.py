from tkinter import Image
from kivy.metrics import inch
from matplotlib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class PDFReport:
    def __init__(self, pdf_output_path):
        self.output_path = pdf_output_path
        self.elements = []
        self.styles = getSampleStyleSheet()

    def add_title_page(self, logo_path, project_title, authors):
        self.elements.append(Image(logo_path, 8 * inch, 4 * inch))
        self.elements.append(Spacer(1, 30))
        self.elements.append(Paragraph("Project Report", self.styles['Title']))
        self.elements.append(Spacer(1, 30))
        table_data = [['Name', 'Roll No.']] + authors
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ])
        table = Table(table_data)
        table.setStyle(table_style)
        self.elements.append(table)
        self.elements.append(PageBreak())

    def add_paragraph(self, text, style='Normal'):
        self.elements.append(Paragraph(text, self.styles[style]))
        self.elements.append(Spacer(1, 12))

    def add_image(self, img_path, caption, width=5*inch, height=7*inch):
        img = Image(img_path)
        img._restrictSize(width, height)
        self.elements.append(img)
        self.elements.append(Paragraph(f"<para align=center>{caption}</para>", self.styles['Heading4']))
        self.elements.append(Spacer(1, 12))

    def build_report(self):
        doc = SimpleDocTemplate(self.output_path, pagesize=A3)
        doc.build(self.elements)
