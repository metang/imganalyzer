content = open('D:/Code/imganalyzer/scripts/diagnose_sunset2.py').read()
content = content.replace('embedding_vector', 'vector')
open('D:/Code/imganalyzer/scripts/diagnose_sunset2.py', 'w').write(content)
print('done')
