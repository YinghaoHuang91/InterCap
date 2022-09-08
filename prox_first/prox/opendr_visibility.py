
def get_vis_():
    import trimesh
    import numpy as np
    import chumpy as ch
    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer, BoundaryRenderer, DepthRenderer
    import pickle as pkl

    v = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    f = np.array([[0, 1, 2]])

    rn = ColoredRenderer()
    rn.camera = ProjectPoints(v=v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([918.457763671875, 918.4373779296875]), c=ch.array([956.9661865234375, 555.944580078125]), k=ch.zeros(5))
    rn.frustum = {'near': 0.01, 'far': 100., 'width': 1920, 'height': 1080}
    rn.v = v
    rn.f = f
    rn.bgcolor = ch.zeros(3)
    rn.vc = ch.ones_like(v)

    img = rn()
    import cv2
    return img * 255

def get_vis(v, f):
    import chumpy as ch
    import numpy as np
    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer, BoundaryRenderer, DepthRenderer

    rn = ColoredRenderer()
    rn.camera = ProjectPoints(v=v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([918.457763671875, 918.4373779296875]), c=ch.array([956.9661865234375, 555.944580078125]), k=ch.zeros(5))
    rn.frustum = {'near': 0.01, 'far': 100., 'width': 1920, 'height': 1080}
    rn.v = v
    rn.f = f
    rn.bgcolor = ch.zeros(3)
    rn.vc = ch.ones_like(v)

    visible_fidxs = np.unique(rn.visibility_image[rn.visibility_image != 4294967295])
    visible_vidxs = np.unique(rn.f[visible_fidxs].ravel())

    return visible_vidxs


def main():
    get_vis_()

if __name__ == '__main__':
    main()
