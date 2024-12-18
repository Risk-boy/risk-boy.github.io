import datetime
import os

def create_md_file(current_date, problem_number, problem_title):
    
    content = f"""
---
title: "백준 #{problem_number} {problem_title}"
categories:
- BOJ
tags:
- [Algorithm, Coding Test, Python]

date: {current_date}
last_modified_at: {current_date}
---

## :link:문제 링크

<a href="https://www.acmicpc.net/problem/{problem_number}" style="text-decoration:none; color:black; font-weight:bold" target="_blank">[{problem_number} {problem_title}]</a>

## :question:문제 설명



## :pencil2:코드

```python
```

## :memo:풀이
"""
    
    relative_path = "./_posts/problem-solving/boj"

    file_name = f"{current_date}-{problem_number}.md"
    file_path = os.path.join(relative_path, file_name)


    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
        
    print(f"{file_name} 파일 생성")
        
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
problem_number = input("문제 번호: ") 
problem_title = input("문제 제목: ") 
create_md_file(current_date, problem_number, problem_title)
