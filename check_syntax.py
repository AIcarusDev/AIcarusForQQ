import ast, sys
files = [
    r'e:\Aic_forQ\core\src\napcat_handler.py',
    r'e:\Aic_forQ\core\src\lifecycle.py',
    r'e:\Aic_forQ\core\src\llm\core\schema.py',
    r'e:\Aic_forQ\core\src\llm\session.py',
    r'e:\Aic_forQ\core\src\llm\core\llm_core.py',
    r'e:\Aic_forQ\core\src\llm\core\retry.py',
    r'e:\Aic_forQ\core\src\llm\core\provider.py',
    r'e:\Aic_forQ\core\src\llm\IS\core.py',
]
ok = True
for f in files:
    try:
        ast.parse(open(f, encoding='utf-8').read())
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'ERROR: {f}: {e}')
        ok = False
sys.exit(0 if ok else 1)
