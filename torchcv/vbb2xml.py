import os, glob, argparse
import cv2
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from lxml import etree, objectify

def vbb_anno2dict(vbb_file, cam_id):
    """
    Parse caltech vbb annotation file to dict
    Args:
        vbb_file: input vbb file path
        cam_id: camera id
        person_types: list of person type that will be used (total 4 types: person, person-fa, person?, people).
            If None, all will be used:
    Return:
        Annotation info dict with filename as key and anno info as value
    """

    filename = os.path.splitext(os.path.basename(vbb_file))[0]
    annos = defaultdict(dict)
    vbb = loadmat(vbb_file)
    

    # object info in each frame: id, pos, occlusion, lock, posv
    objLists = vbb['A'][0][0][1][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    

    # person index
    person_types = ['person', 'cyclist', 'people', 'person?']
    person_index_list = [x for x in range(len(objLbl)) if objLbl[x] in person_types]


    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:

            frame_name = str(cam_id) + "/" + str(filename) + "/" + 'I{:05d}'.format(frame_id)
            
            annos[frame_name] = defaultdict(list)
            annos[frame_name]["id"] = frame_name
            
            try :
                for fid, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                    fid = int(fid[0][0]) - 1  # for matlab start from 1 not 0
                    if not fid in person_index_list:  # only use bbox whose label is given person type
                        continue
                    annos[frame_name]["label"] = objLbl[fid]
                    pos = pos[0].tolist()
                    occl = int(occl[0][0])
                    annos[frame_name]["occlusion"].append(occl)
                    annos[frame_name]["bbox"].append(pos)
                # if not annos[frame_name]["bbox"]:
                #     del annos[frame_name]
            except :
                import pdb
                pdb.set_trace 

    return annos

def instance2xml_base(anno, bbox_type='xyxy'):
    """
    Parse annotation data to VOC XML format
    Args:
        anno: annotation info returned by vbb_anno2dict function
        img_size: camera captured image size
        bbox_type: bbox coordinate record format: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)
    Returns:
        Annotation xml info tree
    """
    assert bbox_type in ['xyxy', 'xywh']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('KAIST Multispectral Ped Benchmark'),
        E.filename(anno['id']),
        E.source(
            E.database('KAIST pedestrian'),
            E.annotation('KAIST pedestrian'),
            E.image('KAIST pedestrian'),
            E.url('https://soonminhwang.github.io/rgbt-ped-detection/')
        ),
        E.size(
            E.width(640),
            E.height(512),
            E.depth(4)
        ),
        E.segmented(0),
    )
    for index, bbox in enumerate(anno['bbox']):
        bbox = [float(x) for x in bbox]
        if bbox_type == 'xyxy':
            xmin, ymin, w, h = bbox
            xmax = xmin+w
            ymax = ymin+h
        else:
            xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        if xmin < 0:
            xmin = 0
        if xmax > 640 - 1:
            xmax = 640 - 1
        if ymin < 0:
            ymin = 0
        if ymax > 512 - 1:
            ymax = 512 - 1
        if ymax <= ymin or xmax <= xmin:
            continue
        E = objectify.ElementMaker(annotate=False)

        anno_tree.append(
            E.object(
            E.name(anno['label']),
            E.bndbox(
                E.x(xmin),
                E.y(ymin),
                E.w(xmax-xmin),
                E.h(ymax-ymin)
            ),
            E.pose('unknown'), 
            E.truncated(0),
            E.difficult(0),
            E.occlusion(anno["occlusion"][index])
            )
        )
    return anno_tree

def parse_anno_file(vbb_inputdir,vbb_outputdir):
    """
    Parse Caltech data stored in seq and vbb files to VOC xml format
    Args:
        vbb_inputdir: vbb file saved pth
        vbb_outputdir: vbb data converted xml file saved path
        person_types: list of person type that will be used (total 4 types: person, person-fa, person?, people).
            If None, all will be used:
    """
    # annotation sub-directories in hda annotation input directory

    assert os.path.exists(vbb_inputdir)
    sub_dirs = os.listdir(vbb_inputdir)
    for sub_dir in sub_dirs:
        print("Parsing annotations of camera: ", sub_dir)
        cam_id = sub_dir
        vbb_files = glob.glob(os.path.join(vbb_inputdir, sub_dir, "*.vbb"))
        for vbb_file in vbb_files:
            annos = vbb_anno2dict(vbb_file, cam_id)
            if annos:

                vbb_outdir = os.path.join(vbb_outputdir, sub_dir)

                if not os.path.exists(vbb_outdir):
                    os.makedirs(vbb_outdir)

                for filename, anno in sorted(annos.items(), key=lambda x: x[0]):
                    # if "bbox" in anno:
                    anno_tree = instance2xml_base(anno)

                    out_dir = vbb_outdir + '/' + filename[6:10]

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)   

                    outfile = os.path.join(out_dir, os.path.splitext(filename[11:])[0]+".xml")

                    print("Generating annotation xml file of picture: ", filename)
                    etree.ElementTree(anno_tree).write(outfile, pretty_print=True)    

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("vbb_dir", help="Caltech dataset vbb data root directory")
    # parser.add_argument("output_dir", help="Root saving path for frame and annotation files")
    # parser.add_argument("person_type", default="person", type=str, help="Person type extracted within 4 options: "
    #                                                   "'person', 'person-fa', 'person?', 'people'. If multiple type used,"
    #                                                   "separated with comma",choices=["person", "person-fa", "person?", "people"])

    # import pdb
    # pdb.set_trace()

    vbb_dir = './datasets/kaist-rgbt/annotations_vbb'
    outdir = './datasets/kaist-rgbt'
    anno_out = os.path.join(outdir, "annotations-xml-15")
    parse_anno_file(vbb_dir, anno_out)
    print("Generating done!")

if __name__ == "__main__":
    main()