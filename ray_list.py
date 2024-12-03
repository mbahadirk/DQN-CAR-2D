from ray import Ray


def create_rays(surface):
    ray_list = [
        Ray(0, surface, name='frontMidRay'),
        Ray(30, surface, name='frontRightRay'),
        Ray(-30, surface, name='frontLeftRay'),
        Ray(-90, surface, name='midLeftRay'),
        Ray(90, surface, name='midRightRay'),
        Ray(180, surface, name='backMidRay'),
        Ray(-150, surface, name='backLeftRay'),
        Ray(150, surface, name='backRightRay')
    ]
    return ray_list