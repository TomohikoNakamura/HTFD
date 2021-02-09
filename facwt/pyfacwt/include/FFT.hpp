/*!
 * @file FFT.hpp
 * @brief Wrapper of fftw3
 * @date 5. Jul. 2015.
 * @author Tomohiko Nakamura
 */

#pragma once

#include <fftw3.h>
#include <cmath>
#include <complex>

/*! @class FFT
 * @brief FFT for complex signal
 */
class FFT {
 protected:
  //! @brief fftwf configuration
  fftwf_plan plan;
  /*! @brief initialization
   * @param D waveform length
   */
  void init_first(const int &D) {
    assert(D > 1);
    size = D;
    spec = new std::complex<float>[D];
    wave = new std::complex<float>[D];
    for (int d = 0; d < D; d++) {
      spec[d] = std::complex<float>(0.0,0.0);
      wave[d] = std::complex<float>(0.0,0.0);
    }
  }

 public:
  //! @brief spectrum
  std::complex<float> *spec;
  //! @brief waveform
  std::complex<float> *wave;
  //! @brief waveform length
  int size;
  //! @brief constructor
  FFT() {
    spec = nullptr;
    wave = nullptr;
    plan = nullptr;
    size = -1;
  }
  //! @brief destructor
  ~FFT() {
    if (spec != nullptr) {
      delete[] spec;
    }
    if (wave != nullptr) {
      delete[] wave;
    }
    if (plan != nullptr) {
      fftwf_destroy_plan(plan);
    };
  }
  //! @brief perform FFT
  void perf() { fftwf_execute(plan); }
};

/// For complex signal
/*! @class ForwardFFT
 * @brief Forward FFT for complex signal
 */
class ForwardFFT : public FFT {
 public:
  /*! @brief initialization
   * @param D waveform length
   * @param flags FFTW_FLAGS
   */
  void init(const int &D, unsigned flags) {
    init_first(D);
    plan = fftwf_plan_dft_1d(D, reinterpret_cast<fftwf_complex *>(wave),
                             reinterpret_cast<fftwf_complex *>(spec),
                             FFTW_FORWARD, flags);
  }
};

/*! @class BackwardFFT
 * @brief Inverse FFT for complex signal
 */
class BackwardFFT : public FFT {
 public:
  /*! @brief initialization
   * @param D waveform length
   * @param flags FFTW_FLAGS
   */
  void init(const int &D, unsigned flags) {
    init_first(D);
    plan = fftwf_plan_dft_1d(D, reinterpret_cast<fftwf_complex *>(spec),
                             reinterpret_cast<fftwf_complex *>(wave),
                             FFTW_BACKWARD, flags);
  }
};

/// For real signal
/*! @class realFFT
 * @brief FFT for real signal
 */
class realFFT {
 protected:
  //! @brief fftwf configuration
  fftwf_plan plan;
  /*! @brief initialization
   * @param D waveform length
   */
  void init_first(const int &D) {
    size = D;
    spec = new std::complex<float>[ D / 2 + 1 ];
    wave = new float[D];
  }

 public:
  //! @brief spectrum
  std::complex<float> *spec;
  //! @brief waveform
  float *wave;
  //! @brief waveform length
  int size;
  //! @brief constructor
  realFFT() {
    spec = nullptr;
    wave = nullptr;
    plan = nullptr;
    size = -1;
  }
  //! @brief destructor
  ~realFFT() {
    if (spec != nullptr) {
      delete[] spec;
    }
    if (wave != nullptr) {
      delete[] wave;
    }
    if (plan != nullptr) {
      fftwf_destroy_plan(plan);
    };
  }
  //! @brief perform FFT
  void perf() {
    assert(plan != NULL && plan != nullptr);
    fftwf_execute(plan);
  }
};

/*! @class ForwardrealFFT
 * @brief Forward FFT for real signal
 */
class ForwardrealFFT : public realFFT {
 public:
  /*! @brief initialization
   * @param D waveform length
   * @param flags FFTW_FLAGS
   */
  void init(const int &D, unsigned flags) {
    init_first(D);
    plan = fftwf_plan_dft_r2c_1d(
        D, wave, reinterpret_cast<fftwf_complex *>(spec), flags);
  }
};

/*! @class BackwardrealFFT
 * @brief Inverse FFT for real signal
 */
class BackwardrealFFT : public realFFT {
 public:
  /*! @brief initialization
   * @param D waveform length
   * @param flags FFTW_FLAGS
   */
  void init(const int &D, unsigned flags) {
    init_first(D);
    plan = fftwf_plan_dft_c2r_1d(D, reinterpret_cast<fftwf_complex *>(spec),
                                 wave, flags);
  }
};
