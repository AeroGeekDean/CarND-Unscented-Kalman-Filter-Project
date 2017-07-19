#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  is_initialized_ = false;  // one shot initialization
  n_x_ = 5;  // state dimension
  n_aug_ = 7;  // augmented state dimension
  lambda_ = 3 - n_x_; // sigma point spreading parameter
  weights_ = VectorXd(2*n_aug_+1); // weights for sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1); // predicted sigma points matrix

  weights_.fill(0.0);
  Xsig_pred_.fill(0.0);
  time_us_ = 0.0;

  use_laser_ = true;  // if this is false, laser measurements will be ignored (except during init)
  use_radar_ = true;  // if this is false, radar measurements will be ignored (except during init)

  // initial state vector
  x_ = VectorXd(n_x_); // full init later during 1-shot init...

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_); // start with identity matrix...

  // --- process noise ---

  // Process noise standard deviation longitudinal acceleration in [m/s^2]
  std_a_ = 9.8*0.125; // using 1/8th of a "G" (ie: gravitational accel)

  // Process noise standard deviation yaw acceleration in [rad/s^2]
  std_yawdd_ = 3.14*0.25; // 45 deg/s^2

  // --- measurement noise ---
  std_laspx_ = 0.15;  // Laser measurement noise standard deviation position1 in [m]
  std_laspy_ = 0.15;  // Laser measurement noise standard deviation position2 in [m]
  std_radr_ = 0.3;  // Radar measurement noise standard deviation radius in [m]
  std_radphi_ = 0.03;  // Radar measurement noise standard deviation angle in [rad]
  std_radrd_ = 0.3;  // Radar measurement noise standard deviation radius change in [m/s]
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  // === Initialization ===
  if (!is_initialized_)
  {
    // Initialize state.
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      cout << "R:" << endl;
      float rho = meas_package.raw_measurements_[0];
      float theta = meas_package.raw_measurements_[1];
      float rhod = meas_package.raw_measurements_[2];
      x_ << rho*cos(theta), // Px
            rho*sin(theta), // Py
            0, 0, 0; // v, psi, psid
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      cout << "L:" << endl;
      x_ << meas_package.raw_measurements_[0], // Px
            meas_package.raw_measurements_[1], // Py
            0, 0, 0; // v, psi, psid
    }

    cout << "x_(init) = " << x_ << endl;

    time_us_ = meas_package.timestamp_;
    nis_laser_ = 0.0;
    nis_radar_ = 0.0;
    is_initialized_ = true;
    return;
  }

  // === Normal operation ===
  double dt = (meas_package.timestamp_ - time_us_)/1E6; // [sec]

  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && use_radar_)
  { // NOTE - Skip if use_radar is turned OFF...
    time_us_ = meas_package.timestamp_;
    Prediction(dt);
    nis_radar_ = UpdateRadar(meas_package);
  }
  else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && use_laser_)
  { // NOTE - Skip if use_laser is turned OFF...
    time_us_ = meas_package.timestamp_;
    Prediction(dt);
    nis_laser_ = UpdateLidar(meas_package);
  }

  return;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} dt the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double dt) {

  SigmaPointPrediction(GenerateAugmentedSigmaPoints(), dt);

  PredictMeanAndCovariance(x_, P_);
}


MatrixXd UKF::GenerateSigmaPoints()
{
  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2*n_x_+1);

  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  //calculate sigma points ...
  Xsig.col(0) = x_;
  for (int i=0; i<n_x_; i++)
  {
    Xsig.col(i+1)      = x_ + sqrt(lambda_+n_x_)*A.col(i);
    Xsig.col(i+1+n_x_) = x_ - sqrt(lambda_+n_x_)*A.col(i);
  }
  return Xsig;
}


MatrixXd UKF::GenerateAugmentedSigmaPoints()
{
  VectorXd x_aug = VectorXd(n_aug_); // augmented mean vector
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);  // augmented state covariance
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);  // sigma point matrix

  //create augmented mean state
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_; // set x_ as first n_x_ elements of x_aug_

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i=0; i<n_aug_; i++)
  {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_)*A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_)*A.col(i);
  }

  return Xsig_aug;
}


void UKF::SigmaPointPrediction(const MatrixXd& Xsig_aug, const double& dt)
{
  double epsilon = 0.001; // yawd tolerance for straight vs curved, [rad/s]

  for (int i=0; i<Xsig_aug.cols(); i++)
  {
    //extract values for better readability
    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //(p)redicted state values
    double v_p    = v;
    double yaw_p  = yaw + yawd*dt;
//      while (yaw_p> M_PI) yaw_p-=2.*M_PI;  // this caused issues. commented out.
//      while (yaw_p<-M_PI) yaw_p+=2.*M_PI;
    double yawd_p = yawd;

    double px_p, py_p;
    if (fabs(yawd) > epsilon)  // curved model
    {
      px_p = p_x + v/yawd*(sin(yaw_p)-sin(yaw));
      py_p = p_y + v/yawd*(cos(yaw)-cos(yaw_p));
    } else                     // straight line model
    {
      px_p = p_x + v*dt*cos(yaw);
      py_p = p_y + v*dt*sin(yaw);
    }

    // pre-compute dt^2/2
    double half_dtdt = 0.5*dt*dt;

    //add noise
    px_p   += half_dtdt*cos(yaw)*nu_a;
    py_p   += half_dtdt*sin(yaw)*nu_a;
    v_p    += nu_a*dt;
    yaw_p  += half_dtdt*nu_yawdd;
    yawd_p += nu_yawdd*dt;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  return;
}


void UKF::PredictMeanAndCovariance(VectorXd& x_out, MatrixXd& P_out)
{
  //set weights
  weights_(0) = lambda_/(lambda_+n_aug_);
  for (int i=1; i<Xsig_pred_.cols(); i++)
    weights_(i) = 0.5/(lambda_+n_aug_);

  //predict state mean
  x_out = Xsig_pred_*weights_;

  //predict state covariance matrix
  MatrixXd diff = Xsig_pred_.colwise() - x_out;
    while (diff(3)> M_PI) diff(3)-=2.*M_PI;
    while (diff(3)<-M_PI) diff(3)+=2.*M_PI;

  P_out = ( diff.array().rowwise() * weights_.transpose().array() ).matrix()
          * diff.transpose(); // using vectorization method
  return;
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
float UKF::UpdateLidar(MeasurementPackage meas_package) {

  int n_z = 2;  //set measurement dimension: px, py for laser

  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);  //matrix for sigma points in measurement space
  VectorXd z_pred = VectorXd(n_z);  //mean predicted measurement
  MatrixXd S = MatrixXd(n_z, n_z);  //measurement covariance matrix S
  VectorXd z = meas_package.raw_measurements_;  //actual measurements

  // --- Predict lidar measurement & covariance (z_pred, S) ---

  Zsig.fill(0.0);
  z_pred.fill(0.0);
  S.fill(0.0);

  //transform sigma points into measurement space
  for (int i=0; i<Xsig_pred_.cols(); i++)
  { // trivial for laser...
    Zsig(0,i) = Xsig_pred_(0,i); // px
    Zsig(1,i) = Xsig_pred_(1,i); // py
  }

  //calculate mean predicted measurement
  z_pred = Zsig*weights_;

  //calculate measurement covariance matrix S

  MatrixXd z_diff = Zsig.colwise() - z_pred;

  MatrixXd R = MatrixXd(n_z, n_z);  //measurement noise
  R.fill(0.0);
  R(0,0) = std_laspx_*std_laspx_;
  R(1,1) = std_laspy_*std_laspy_;

  S = (z_diff.array().rowwise() * weights_.transpose().array()).matrix()
      *z_diff.transpose() + R;

  //--- Update Lidar State (x_, P_) ---

  MatrixXd x_diff = Xsig_pred_.colwise() - x_;

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);  //matrix for cross correlation Tc
  Tc = (x_diff.array().rowwise() * weights_.transpose().array()).matrix()
       *z_diff.transpose();

  //calculate Kalman gain K;
  MatrixXd K = MatrixXd(n_x_,n_z);
  K = Tc*S.inverse();

  //update state mean and covariance matrix
  VectorXd z_residual = z - z_pred;

  x_ += K*z_residual;
    while (x_(3)> M_PI) x_(3)-=2.*M_PI;
    while (x_(3)<-M_PI) x_(3)+=2.*M_PI;
  P_ -= K*S*K.transpose();

  // --- Calc NIS ---
  float NIS = z_residual.transpose()*S.inverse()*z_residual;

  return NIS;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
float UKF::UpdateRadar(MeasurementPackage meas_package) {
//  cout << "UKF::UpdateRadar" << endl;

  int n_z = 3;  //set measurement dimension: r, phi, & r_dot for radar

  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);  //matrix for sigma points in measurement space
  VectorXd z_pred = VectorXd(n_z);  //mean predicted measurement
  MatrixXd S = MatrixXd(n_z, n_z);  //measurement covariance matrix S
  VectorXd z = meas_package.raw_measurements_;  //actual measurements


  // --- Predict radar measurement & covariance (z_pred, S) ---

  Zsig.fill(0.0);
  z_pred.fill(0.0);
  S.fill(0.0);

  //transform sigma points into measurement space
  for (int i=0; i<Xsig_pred_.cols(); i++)
  {
    double px   = Xsig_pred_(0,i);
    double py   = Xsig_pred_(1,i);
    double v    = Xsig_pred_(2,i);
    double psi  = Xsig_pred_(3,i);
//    double psid = Xsig_pred(4,i);

    Zsig(0,i) = sqrt(px*px + py*py);
    Zsig(1,i) = atan2(py, px);
    Zsig(2,i) = v*(px*cos(psi) + py*sin(psi)) / Zsig(0,i);
  }

  //calculate mean predicted measurement
  z_pred = Zsig*weights_;

  //calculate measurement covariance matrix S

  MatrixXd z_diff = Zsig.colwise() - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  MatrixXd R = MatrixXd(n_z, n_z);  //measurement noise
  R.fill(0.0);
  R(0,0) = std_radr_*std_radr_;
  R(1,1) = std_radphi_*std_radphi_;
  R(2,2) = std_radrd_*std_radrd_;

  S = (z_diff.array().rowwise() * weights_.transpose().array()).matrix()
      *z_diff.transpose() + R;

  //--- Update Radar State (x_, P_) ---

  MatrixXd x_diff = Xsig_pred_.colwise() - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);  //matrix for cross correlation Tc
  Tc = (x_diff.array().rowwise() * weights_.transpose().array()).matrix()
       *z_diff.transpose();

  //calculate Kalman gain K;
  MatrixXd K = MatrixXd(n_x_,n_z);
  K = Tc*S.inverse();

  //update state mean and covariance matrix
  VectorXd z_residual = z - z_pred;
    while (z_residual(1)> M_PI) z_residual(1)-=2.*M_PI;
    while (z_residual(1)<-M_PI) z_residual(1)+=2.*M_PI;

  x_ += K*z_residual;
    while (x_(3)> M_PI) x_(3)-=2.*M_PI;
    while (x_(3)<-M_PI) x_(3)+=2.*M_PI;
  P_ -= K*S*K.transpose();

  // --- Calc NIS ---
  double NIS = z_residual.transpose()*S.inverse()*z_residual;

  return NIS;
}
