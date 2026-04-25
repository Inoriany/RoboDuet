import re
txt = open('/root/RoboDuet/resources/robots/b2z1/urdf/b2z1.urdf').read()
links = []
for line in txt.splitlines():
    s = line.strip()
    if s.startswith('<link'):
        m = re.search(r'name="([^"]+)"', s)
        links.append(m.group(1) if m else '(unnamed)')
for i, name in enumerate(links):
    print(i, name)
