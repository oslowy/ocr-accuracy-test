def area_within_ratio(observed_poly, target_poly):
    return (observed_poly.intersection(target_poly)).area / observed_poly.area
