from bs4 import BeautifulSoup

def extract_departments_from_html(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # 找出所有 table row 內的科別與病症
    rows = soup.select('table tbody tr')
    output_lines = []

    for row in rows:
        dept_cell = row.find('th')
        desc_cell = row.find('td')
        if not dept_cell or not desc_cell:
            continue

        department = dept_cell.get_text(strip=True)
        description = desc_cell.get_text(separator="", strip=True)

        output_lines.append(f"科別 : {department}")
        output_lines.append(f"科別病症敘述 : {description}")
        output_lines.append("")  # 空行分隔

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

# 用法
extract_departments_from_html("hospital/Index.htm", "hospital/output.txt")
