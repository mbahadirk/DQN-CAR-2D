from ray import Ray


def create_rays(surface):
    ray_list = [
        Ray(-60, surface, name='frontLeftRay5'),
        Ray(-30, surface, name='frontLeftRay3'),

        Ray(0, surface, name='frontMidRay2'),

        Ray(30, surface, name='frontRightRay3'),
        Ray(60, surface, name='frontRightRay5'),
    ]
    return ray_list
