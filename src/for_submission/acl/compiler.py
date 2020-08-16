import os


full_content = []
for file in os.listdir("/Users/kai/Downloads/cleaned_acl_arc_ascii"):
    path = "/Users/kai/Downloads/cleaned_acl_arc_ascii/" + file
    with open(path, encoding='utf-8',errors='ignore') as f:
        content = f.read()
        content = content.replace("\n", " ").replace("- ", "")
    full_content.append(content)

full_text = "\n".join(full_content)
with open("out_file.txt", "w") as f:
    f.write(full_text)
print("done")
