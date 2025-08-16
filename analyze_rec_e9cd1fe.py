with open('rec_e9cd1fe.db', 'rb') as f:
    data = f.read()

# Decodificar como UTF-16
try:
    text = data.decode('utf-16', errors='ignore')
except:
    text = data.decode('utf-16le', errors='ignore')

print(f"=== ANÁLISE DO ARQUIVO rec_e9cd1fe.db ===\n")

# Procurar por definições de tabelas
import re

# Encontrar comandos CREATE TABLE
tables = re.findall(r'CREATE TABLE\s+(\w+)\s*\([^)]+(?:\([^)]*\)[^)]*)*\)', text, re.IGNORECASE | re.DOTALL)

print("🗂️ ESTRUTURA DAS TABELAS ENCONTRADAS:")
print("=" * 60)

table_names = []
for match in re.finditer(r'CREATE TABLE\s+(\w+)\s*\((.*?)\)', text, re.IGNORECASE | re.DOTALL):
    table_name = match.group(1)
    table_def = match.group(2)
    
    # Limpar caracteres de controle
    clean_def = re.sub(r'[^\x20-\x7E\n\r\t]', '', table_def)
    
    print(f"\n📋 Tabela: {table_name}")
    print("   Definição:")
    
    # Dividir em linhas e limpar
    lines = clean_def.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('--'):
            print(f"     {line}")
    
    table_names.append(table_name)

# Procurar por dados inseridos
print(f"\n\n📊 DADOS INSERIDOS:")
print("=" * 60)

for table_name in table_names:
    # Procurar por INSERTs desta tabela
    pattern = rf'INSERT INTO\s+{table_name}\s+.*?VALUES\s*\([^)]+\)'
    inserts = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if inserts:
        print(f"\n📁 Tabela {table_name}: {len(inserts)} registros")
        
        # Mostrar alguns exemplos
        for i, insert in enumerate(inserts[:3], 1):
            clean_insert = re.sub(r'[^\x20-\x7E]', ' ', insert)
            # Extrair apenas os valores
            values_match = re.search(r'VALUES\s*\((.*?)\)', clean_insert, re.IGNORECASE)
            if values_match:
                values = values_match.group(1).strip()
                print(f"     Registro {i}: {values}")
    else:
        print(f"\n📁 Tabela {table_name}: 0 registros")

# Procurar especificamente por dados de perguntas e respostas
print(f"\n\n📝 CONTEÚDO DE PERGUNTAS E RESPOSTAS:")
print("=" * 60)

# Procurar por perguntas
question_pattern = r'pergunta["\']?\s*[,:]?\s*["\']([^"\']+)["\']'
questions = re.findall(question_pattern, text, re.IGNORECASE)

if questions:
    print(f"\n❓ Perguntas encontradas ({len(questions)} total):")
    for i, question in enumerate(questions[:5], 1):
        clean_question = re.sub(r'[^\x20-\x7E]', ' ', question).strip()
        if clean_question:
            print(f"   {i}. {clean_question}")
    if len(questions) > 5:
        print(f"   ... e mais {len(questions) - 5} perguntas")

# Procurar por respostas
answer_pattern = r'resposta["\']?\s*[,:]?\s*["\']([^"\']+)["\']'
answers = re.findall(answer_pattern, text, re.IGNORECASE)

if answers:
    print(f"\n💬 Respostas encontradas ({len(answers)} total):")
    for i, answer in enumerate(answers[:5], 1):
        clean_answer = re.sub(r'[^\x20-\x7E]', ' ', answer).strip()
        if clean_answer:
            print(f"   {i}. {clean_answer[:100]}{'...' if len(clean_answer) > 100 else ''}")
    if len(answers) > 5:
        print(f"   ... e mais {len(answers) - 5} respostas")

# Informações gerais do arquivo
print(f"\n\n📋 INFORMAÇÕES GERAIS:")
print("=" * 60)
print(f"📁 Nome do arquivo: rec_e9cd1fe.db")
print(f"💾 Tamanho: {len(data):,} bytes ({len(data)/1024:.1f} KB)")
print(f"📝 Formato: SQLite em UTF-16")
print(f"🗂️ Número de tabelas: {len(table_names)}")

if table_names:
    print(f"📊 Tabelas: {', '.join(table_names)}")

# Verificar se há outros elementos do banco
indexes = re.findall(r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)', text, re.IGNORECASE)
if indexes:
    print(f"🔍 Índices: {len(indexes)} encontrados ({', '.join(indexes[:5])}{'...' if len(indexes) > 5 else ''})")

# Verificar o cabeçalho SQLite
if text.startswith('SQLite format 3'):
    print(f"✅ Formato SQLite válido detectado")
else:
    print(f"⚠️ Formato SQLite não detectado no início")

# Comparar com o arquivo anterior
print(f"\n\n🔄 COMPARAÇÃO COM rec_4210564.db:")
print("=" * 60)
print(f"rec_4210564.db: 538.1 KB - Estrutura vazia")
print(f"rec_e9cd1fe.db: {len(data)/1024:.1f} KB - {'Contém dados' if (questions or answers) else 'Estrutura vazia'}")

print(f"\n{'='*60}")
print("Análise concluída!")
