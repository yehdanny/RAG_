from bs4 import BeautifulSoup

# 讀取原始 HTML-like txt 檔
with open("rag_v2/data.txt", "r", encoding="utf-8") as f:
    content = f.read()

soup = BeautifulSoup(content, "html.parser")

# 尋找所有的 <tr> 列，這些列通常代表一筆問答紀錄
rows = soup.find_all("tr")

qa_pairs = []

for row in rows:
    cols = row.find_all("td")
    if len(cols) >= 2:
        question_td = cols[0].get_text(strip=True)
        answer_td = cols[1].get_text(separator="\n", strip=True)
        if question_td and answer_td:
            qa_pairs.append((question_td, answer_td))

# 🔸 只保留從第22筆開始的問答對（Q22 之後）
qa_pairs = qa_pairs[21:]  # 注意：list 是從 index 0 開始的

# 寫入 txt 檔案
output_lines = []
for i, (q, a) in enumerate(qa_pairs, start=22):  # 從 Q22 開始編號
    output_lines.append(f"Q{i}: {q}")
    output_lines.append(f"A{i}: {a}")
    output_lines.append("")

# 寫入 output.txt
with open("rag_v2/output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("✅ 問答寫入 output.txt")
