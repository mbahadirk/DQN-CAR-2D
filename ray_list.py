from ray import Ray


def create_rays(surface):
    ray_list = [
        Ray(0, surface, name='frontMidRay2'),

        # Ray(10, surface, name='frontRightRay'),
        # Ray(15, surface, name='frontRightRay2'),
        Ray(30, surface, name='frontRightRay3'),
        # Ray(45, surface, name='frontRightRay4'),
        # Ray(60, surface, name='frontRightRay5'),

        # Ray(-10, surface, name='frontLeftRay'),
        # Ray(-15, surface, name='frontLeftRay2'),
        Ray(-30, surface, name='frontLeftRay3'),
        # Ray(-45, surface, name='frontLeftRay4'),
        # Ray(-60, surface, name='frontLeftRay5'),

        # Ray(-90, surface, name='midLeftRay'),
        # Ray(90, surface, name='midRightRay'8),
        # Ray(180, surface, name='backMidRay'),
        # Ray(-150, surface, name='backLeftRay'),
        # Ray(150, surface, name='backRightRay')
    ]
    return ray_list