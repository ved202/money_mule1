import os
import io
import glob

replacements = {
    '-': '-', '->': '->', 'x': 'x', 'in': 'in', '*': '*', 'alpha': 'alpha', 'sigma': 'sigma', 'Sigma': 'Sigma', '-': '-', '<-': '<-'
}

for root, _, files in os.walk('c:\\Users\\vedan\\Downloads\\mule_detection-main (1)\\mule_detection-main'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            try:
                with io.open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                changed = False
                for k, v in replacements.items():
                    if k in text:
                        text = text.replace(k, v)
                        changed = True
                
                if changed:
                    with io.open(path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"Cleaned {path}")
            except Exception as e:
                pass
