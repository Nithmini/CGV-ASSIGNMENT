import re

def format_text(text):
    """Format the OCR extracted text into a structured receipt format."""
    # Remove any leading/trailing whitespace and fix multiple spaces
    text = re.sub(r'\s+', ' ', text.strip())

    # Ensure proper spacing around symbols like hyphens and dashes
    text = re.sub(r'\s*-\s*', ' - ', text)
    text = re.sub(r'\s*\|\s*', ' | ', text)

    # Insert line breaks before specific headings
    text = re.sub(r'(RESTAURANT|Lorem Ipsum|City Index|Tel:|"TAX INVOICE|CASH RECEIPT)', r'\n\1', text)

    # Insert line breaks after specific keywords to structure the receipt
    text = re.sub(r'(CASH RECEIPT|Sub Total|Cash|Change|Total)', r'\n\1', text)

    # Insert line breaks after each item with a price
    text = re.sub(r'(\d+\.\d{2})', r'\1\n', text)

    # Correct the "Name Qty Total" line
    text = re.sub(r'(Name\s+Qty\s+Total)', r'\n\1\n', text)

    # Split the text into lines for further formatting
    lines = text.splitlines()
    formatted_lines = []

    for line in lines:
        line = line.strip()

        # Handle lines that are likely items with prices
        if re.search(r'\d+\.\d{2}', line):
            parts = re.split(r'(\d+\.\d{2})', line)
            item_part = parts[0].strip()
            price_part = parts[1].strip()

            # Split item part into name and quantity if both are present
            item_match = re.match(r'(.+?)\s+(\d+)\s*$', item_part)
            if item_match:
                name = item_match.group(1).strip()
                qty = item_match.group(2).strip()
                formatted_lines.append(f"{name:<25} {qty:>3}  {price_part:>7}")
            else:
                formatted_lines.append(f"{item_part:<25}      {price_part:>7}")
        else:
            # Handle lines where the price is missing or misaligned (e.g., CheeseBurger)
            if re.match(r'.*\d+\s+\d+$', line):
                parts = re.split(r'\s+', line)
                name = ' '.join(parts[:-2]).strip()
                qty = parts[-2].strip()
                price = float(parts[-1].strip())
                formatted_lines.append(f"{name:<25} {qty:>3}  {price:>7.2f}")
            else:
                # Add line breaks before main labels and format other lines
                if re.match(r'(RESTAURANT|CASH RECEIPT|Sub Total|Cash|Change)', line):
                    formatted_lines.append(f"\n{line}")
                else:
                    formatted_lines.append(line)

    # Add space before the item list and after the totals
    formatted_text = "\n".join(formatted_lines)
    formatted_text = re.sub(r'(CASH RECEIPT\n)', r'\1\n', formatted_text)  # Space after CASH RECEIPT
    formatted_text = re.sub(r'(Change\s+\d+\.\d{2})', r'\1\n\n', formatted_text)  # Space after Change

    return formatted_text

def print_formatted_text(text):
    """Print the formatted text."""
    formatted_text = format_text(text)
    print("Formatted Receipt:\n")
    print(formatted_text)