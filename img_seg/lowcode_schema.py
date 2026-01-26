import uuid

def make_id(prefix="c"):
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def component_info_to_map_and_tree(component_info_list):
    """
    Convert imgseg component_info (list of dicts) to componentsMap and componentsTree.
    Simple rules:
      - generate id for each component
      - use left_top_coord, width, height as bbox
      - if an entry has 'children', recursively process them and set parent
    Returns: (componentsMap, componentsTree, page_meta)
    """
    components_map = {}
    tree_nodes = []

    def process_item(item, parent_id=None):
        cid = make_id()
        left_top = item.get("left_top_coord") or item.get("left_top_coord") or item.get("left_top_coord", [0,0])
        # left_top might be [x,y] or nested; ensure tuple
        if isinstance(left_top, list) and len(left_top) >= 2:
            x, y = int(left_top[0]), int(left_top[1])
        else:
            x, y = 0, 0
        w = int(item.get("width", 0))
        h = int(item.get("height", 0))
        comp_type = item.get("component_type", "Image")
        props = {}
        if "text" in item:
            props["text"] = item["text"]
        style = {}
        if "rgb_mean" in item:
            style["rgb_mean"] = item["rgb_mean"]
        metadata = {}
        if "img_component_base64" in item:
            metadata["crop_base64"] = item["img_component_base64"]

        components_map[cid] = {
            "id": cid,
            "type": comp_type,
            "parent": parent_id,
            "bbox": [x, y, w, h],
            "props": props,
            "style": style,
            "metadata": metadata,
        }
        # children
        children = []
        if "children" in item and isinstance(item["children"], list):
            for child in item["children"]:
                child_id = process_item(child, cid)
                children.append(child_id)
        # also support nested single 'children' inserted by some APIs
        components_map[cid]["children"] = children
        return cid

    # top-level: some APIs return a father item (list with one father). Handle accordingly.
    root_ids = []
    for info in component_info_list:
        # if info has 'left_top_coord' it's a normal component; if it's wrapper with children, still process
        root_id = process_item(info, None)
        root_ids.append(root_id)

    # build tree structure list
    def build_subtree(cid):
        node = components_map[cid].copy()
        node_children = []
        for child_id in node.get("children", []):
            node_children.append(build_subtree(child_id))
        node["children"] = node_children
        return node

    components_tree = [build_subtree(rid) for rid in root_ids]

    # page meta: not available here; caller can fill if needed
    page_meta = {}
    return components_map, components_tree, page_meta





