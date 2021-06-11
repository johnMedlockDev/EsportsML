mystr = """
position_bot  
position_jng  
position_mid  
position_sup  
position_top
"""

mystr = mystr.split("\n")

for s in mystr:
    print(f'"{s.strip()}",')
