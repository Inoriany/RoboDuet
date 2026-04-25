import xml.etree.ElementTree as ET
tree = ET.parse('/root/RoboDuet/resources/robots/b2z1/urdf/b2z1.urdf')
root = tree.getroot()
# Find joints involving gripperMover or ee_gripper_link
for joint in root.iter('joint'):
    parent = joint.find('parent')
    child  = joint.find('child')
    if parent is None or child is None:
        continue
    p = parent.attrib.get('link','')
    c = child.attrib.get('link','')
    if 'gripper' in p.lower() or 'gripper' in c.lower() or 'ee_' in c.lower():
        origin = joint.find('origin')
        xyz = origin.attrib.get('xyz','N/A') if origin is not None else 'N/A'
        rpy = origin.attrib.get('rpy','N/A') if origin is not None else 'N/A'
        print(f"joint '{joint.attrib.get('name','')}': {p} -> {c}")
        print(f"  origin xyz={xyz}  rpy={rpy}")
