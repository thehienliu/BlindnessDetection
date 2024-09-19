# Get context script
import json
import requests
from bs4 import BeautifulSoup

# Get html content from URL
url = "https://www.eye7.in/retina/diabetic-retinopathy/complete-guide/"
respone = requests.get(url)
text_content = ""

if respone.status_code == 200:

    soup = BeautifulSoup(respone.text, "html")

    # Select the useful content
    heading = ", ".join([f"h{i}.wp-block-heading" for i in range(1, 5)])
    paragraph = "div .gb-inside-container p"
    list_content = "div .gb-inside-container li"
    css_selected = ", ".join([heading, paragraph, list_content])

    selected_elements = soup.select(css_selected)
    title = [i.get_text(strip=True) for i in soup.select(heading)]

    # Add special characters to the heading to create splitting pattern
    contexts = "\n".join(
        [
            (
                e.get_text(strip=True)
                if e.get_text(strip=True) not in title
                else f"### Paragraph: " + e.get_text(strip=True)
            )
            for idx, e in enumerate(selected_elements)
        ]
    )

    # Split the content by heading and remove None
    clean_context = [c for c in contexts.split("### Paragraph: ") if c.strip()]

    with open("diabetic_disease_context.json", "w") as outfile:
        json.dump(clean_context, outfile)
