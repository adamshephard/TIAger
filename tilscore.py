from utils import dist_to_px, get_mask_area, write_json, timing
from nms import slide_nms, to_wsd


@timing
def create_til_score(image_path, xml_path, bulk_path, output_path):
    """slide_level_nms"""
    points = slide_nms(image_path, xml_path, 2048)
    wsd_points = to_wsd(points)
    print(len(wsd_points))

    """Compute TIL score and write to output"""
    til_area = dist_to_px(8, 0.5) ** 2
    tils_area = len(wsd_points) * til_area
    stroma_area = get_mask_area(bulk_path)
    tilscore = int((100 / int(stroma_area)) * int(tils_area))
    tilscore = min(100, tilscore)
    tilscore = max(0, tilscore) 
    print(f"tilscore = {tilscore}")
    write_json(tilscore, output_path)