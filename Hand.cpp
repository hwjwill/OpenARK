#include "stdafx.h"
#include "version.h"

#include "Hand.h"

#include "Util.h"
#include "Visualizer.h"

namespace ark {
    Hand::Hand() {
    }

    Hand::Hand(Vec3f centroid_xyz, Point2i centroid_ij, std::vector<Vec3f> fingers_xyz,
        std::vector<Point2i> fingers_ij, std::vector<Vec3f> defects_xyz, std::vector<Point2i> defects_ij,
        double svm_confidence)
    {
        this->center_xyz = centroid_xyz;
        this->center_ij = centroid_ij;
        this->fingers_xyz = fingers_xyz;
        this->fingers_ij = fingers_ij;
        this->defects_xyz = defects_xyz;
        this->defects_ij = defects_ij;
        this->svm_confidence = svm_confidence;
    }

    Hand::~Hand() { }

    int Hand::numFingers() const {
        return (int)fingers_xyz.size();
    }

    bool Hand::touchObject(std::vector<double> &equation, const double threshold) const
    {
        if (equation.size() == 0)
        {
            return false;
        }

        for (int i = 0; i < fingers_xyz.size(); i++)
        {
            double x = fingers_xyz[i][0];
            double y = fingers_xyz[i][1];
            double z = fingers_xyz[i][2];

            if (z == 0)
            {
                return false;
            }

            double z_hat = equation[0] * x + equation[1] * y + equation[2];
            double r_squared = (z - z_hat) * (z - z_hat);

            if (r_squared < threshold)
            {
                return true;
            }
        }

        return false;
    }
}