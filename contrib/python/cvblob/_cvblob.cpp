#include <boost/python.hpp>
#include <iostream>
#include <opencv/cv.h>
#include <cvblob.h> 
#include "pyboostcvconverter/pyboostcvconverter.hpp"

namespace bp = boost::python;
using namespace bp;

struct Blob : cvb::CvBlob { };
struct Track : cvb::CvTrack { };

#define tuple2CvPoint(a) cvPoint(bp::extract<int>(a[0]), bp::extract<int>(a[1]))
#define tuple2CvPoint2D64f(a) cvPoint2D64f(bp::extract<double>(a[0]), bp::extract<double>(a[1]))
#define tuple2CvScalar(a) cvScalar(bp::extract<double>(a[0]), bp::extract<double>(a[1]), bp::extract<double>(a[2]))

//simple casting to our wrapped structs
#define object2CvBlob(a) (cvb::CvBlob*)bp::extract<Blob*>(a)
#define object2CvTrack(a) (cvb::CvTrack*)bp::extract<Track*>(a)

cvb::CvChainCodes* list2CvChainCodes(bp::list l) {
  cvb::CvChainCodes *cc = new cvb::CvChainCodes();
  for (int i = 0; i < len(l); i++) {
    cc->push_back((cvb::CvChainCode) bp::extract<int>(l[i]));
  }
  return cc; 
}

int getIPL_DEPTH_LABEL(){ 
    return IPL_DEPTH_LABEL;
}
namespace cvb {
  int getCV_BLOB_MAX_LABEL(){ 
    return CV_BLOB_MAX_LABEL;
  }
}

//class wrapper for the ContourChainCode
//uses Python-based types 
class ContourChainCode {
  public: 
  tuple startingPoint;
  bp::list chainCode;  

  cvb::CvContourChainCode* toCvContourChainCode();
};


//function to convert to a cvBlob type cvContourChainCode
cvb::CvContourChainCode* ContourChainCode::toCvContourChainCode(){
  cvb::CvContourChainCode *ccc = new cvb::CvContourChainCode;
  ccc->startingPoint = tuple2CvPoint(startingPoint);
  cvb::CvChainCodes *cc = list2CvChainCodes(chainCode);
  ccc->chainCode = *cc;
  
  return ccc; 
}

//function to convert a python list of ContourChainCode objects
//to a CvContoursChainCode object 
cvb::CvContoursChainCode* list2CvContoursChainCode(bp::list l) {
  cvb::CvContoursChainCode *ccc = new cvb::CvContoursChainCode();
  for (int i = 0; i < len(l); i++) {
    ContourChainCode c = extract<ContourChainCode>(l[i]);
    ccc->push_back(c.toCvContourChainCode());
  }
  
  return ccc;
}

class BlobTracker {
    private:
    cvb::CvBlobs blobs;
    cvb::CvTracks tracks;

    public:
    void process(const bp::object &img_mask) {
        IplImage* segmentated = new IplImage(pbcvt::fromNDArrayToMat(img_mask.ptr()));
  
        IplConvKernel* morphKernel = cvCreateStructuringElementEx(5, 5, 1, 1, CV_SHAPE_RECT, NULL);
        cvMorphologyEx(segmentated, segmentated, NULL, morphKernel, CV_MOP_OPEN, 1);

        IplImage* labelImg = cvCreateImage(cvGetSize(segmentated), IPL_DEPTH_LABEL, 1);

        cvReleaseBlobs(blobs);
        blobs = cvb::CvBlobs();
        unsigned int result = cvb::cvLabel(segmentated, labelImg, blobs);

        cvb::cvUpdateTracks(blobs, tracks, 200., 5);

        cvReleaseImage(&labelImg);
        delete segmentated;
        cvReleaseStructuringElement(&morphKernel);
    }

    bp::object getBlobs() {
        //return blobs;
        return bp::object();
    }

    bp::dict getTracks() {
        bp::dict py_tracks;

        cvb::CvTracks::iterator it;

        for ( it = tracks.begin(); it != tracks.end(); it++ )
        {
            bp::object py_track((Track *)it->second);
            py_tracks[it->first] = py_track;
        }

        return py_tracks;
    };
};

//boost block -- here's where we reveal everything to Python
BOOST_PYTHON_MODULE(_cvblob) {
    class_<BlobTracker> ("BlobTracker")
        .def("process", &BlobTracker::process)
        .def("getBlobs", &BlobTracker::getBlobs)
        .def("getTracks", &BlobTracker::getTracks);


  class_<ContourChainCode> ("ContourChainCode")
    .def_readwrite("startingPoint", &ContourChainCode::startingPoint) 
    .def_readwrite("chainCode", &ContourChainCode::chainCode);

  class_<Blob> ("Blob")
    .def_readwrite("label", &Blob::label)
    .def_readwrite("minx", &Blob::minx)
    .def_readwrite("maxx", &Blob::maxx)
    .def_readwrite("miny", &Blob::miny)
    .def_readwrite("maxy", &Blob::maxy)
    .def_readwrite("centroid", &Blob::centroid)
    .def_readwrite("m00", &Blob::m00)
    .def_readwrite("area", &Blob::area)
    .def_readwrite("m10", &Blob::m10)
    .def_readwrite("m01", &Blob::m01)
    .def_readwrite("m11", &Blob::m11)
    .def_readwrite("m20", &Blob::m20)
    .def_readwrite("m02", &Blob::m02)
    .def_readwrite("u11", &Blob::u11)
    .def_readwrite("u20", &Blob::u20)
    .def_readwrite("u02", &Blob::u02)
    .def_readwrite("n11", &Blob::n11)
    .def_readwrite("n20", &Blob::n20)
    .def_readwrite("n02", &Blob::n02)
    .def_readwrite("p1", &Blob::p1)
    .def_readwrite("p2", &Blob::p2)
    .def_readwrite("contour", &Blob::contour)
    .def_readwrite("internalContours", &Blob::internalContours);
 
  class_<Track> ("Track")
    .def_readwrite("id", &Track::id)
    .def_readwrite("label", &Track::label)
    .def_readwrite("minx", &Track::minx)
    .def_readwrite("maxx", &Track::maxx)
    .def_readwrite("miny", &Track::miny)
    .def_readwrite("maxy", &Track::maxy)
    .def_readwrite("centroid", &Track::centroid)
    .def_readwrite("lifetime", &Track::lifetime)
    .def_readwrite("active", &Track::active)
    .def_readwrite("inactive", &Track::inactive);
  
  def("getIPL_DEPTH_LABEL", getIPL_DEPTH_LABEL); 
  def("getCV_BLOB_MAX_LABEL", cvb::getCV_BLOB_MAX_LABEL);
}