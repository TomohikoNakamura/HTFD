/*!
 * @file common.hpp
 * @brief common header for time frequency transforms
 * @date 3 Mar. 2015.
 * @author Tomohiko Nakamura
 */

#pragma once

/// c libraries
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cfloat>
#include <sys/time.h>

/// c++ libraries
#include <iostream>
#include <complex>
#include <tuple>
#include <map>
#include <set>
#include <vector>
#include <functional>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

/// my libraries
#include <FFT.hpp>

/*! @namespace utils
 * @brief utility and IO functions
 */
namespace utils
{
  /// constants
  //!@brief float epsilon
  constexpr float EPS = 1.0e-10;
  //!@brief complex zero
  constexpr std::complex<float> C_ZERO = std::complex<float>(0., 0.);

  /*! @class NotImplementedError
   * @brief Exception for not implemented functions
   */
  class NotImplementedError : public std::logic_error
  {
  public:
    NotImplementedError() : std::logic_error("Not Implemented") {}
  };

  // Timer
  /*! @brief Simple timer class
   * @class Timer
   */
  class Timer
  {
  protected:
    //!@brief start time
    timeval startTime;
    //!@brief stop time
    timeval stopTime;
    /*! @brief convert timeval instance into double time [sec]
     * @param timeVal timeval instance
     */
    double get_sec(const timeval &timeVal)
    {
      return timeVal.tv_sec + timeVal.tv_usec * 1.0e-6;
    }

  public:
    //! @brief constructor
    Timer() { ; }
    //! @brief destructor
    ~Timer() { ; }
    //! @brief start timer
    void start() { gettimeofday(&startTime, NULL); }
    /*! @brief stop timer
     * @return elapsed time from the time when "start" function was called
     */
    const double stop()
    {
      gettimeofday(&stopTime, NULL);
      return get_sec(stopTime) - get_sec(startTime);
    }
  };

  /*! @brief Exponent of next higher power of 2
   * @tparam D_TYPE data type
   * @param value a value
   * @return the exponents for the smallest powers of 2
   */
  template <typename D_TYPE>
  const int nextpow2(const D_TYPE &value)
  {
    return std::ceil(std::log2((double)value));
  }

  /*! @brief replacement of magnitude spectrogram
   * @tparam D_TYPE data type
   * @param spectrogram complex spectrogram whose magnitude will be replaced with magSpectrogram
   * @param magSpectrogram magnitude spectrogram
   * @param fastCompFlag fast computation flag: If true, the function does not use "std::polar".
   */
  template <typename D_TYPE>
  void replace_magnitude(std::vector<std::vector<std::complex<D_TYPE>>> &spectrogram,
                         const std::vector<std::vector<D_TYPE>> &magSpectrogram,
                         const bool fastCompFlag)
  {
    for (int l = 0; l < spectrogram.size(); l++)
    {
      for (int d = 0; d < spectrogram[l].size(); d++)
      {
        if (fastCompFlag)
        {
          spectrogram[l][d] = magSpectrogram[l][d] * (spectrogram[l][d] / (EPS + std::abs(spectrogram[l][d])));
        }
        else
        {
          spectrogram[l][d] = std::polar(magSpectrogram[l][d],
                                         std::arg(spectrogram[l][d]));
        }
      }
    }
  }

  /*! @brief get magnitude spectrogram
   * @tparam D_TYPE data type
   * @param spectrogram complex spectrogram
   * @return magnitude spectrogram
   */
  template <typename D_TYPE>
  std::vector<std::vector<D_TYPE>>
  get_magnitude(const std::vector<std::vector<std::complex<D_TYPE>>> &spectrogram)
  {
    std::vector<std::vector<D_TYPE>> magSpectrogram(spectrogram.size());
    for (int l = 0; l < spectrogram.size(); l++)
    {
      magSpectrogram[l].resize(spectrogram[l].size());
      for (int d = 0; d < spectrogram[l].size(); d++)
      {
        magSpectrogram[l][d] = std::abs(spectrogram[l][d]);
      }
    }
    return magSpectrogram;
  }

  template <typename D_TYPE>
  void copy_spectrogram(std::vector<std::vector<std::complex<D_TYPE>>> &new_spectrogram, const std::vector<std::vector<std::complex<D_TYPE>>> &spectrogram, const bool resizeFlag)
  {
    if (resizeFlag)
    {
      new_spectrogram.resize(spectrogram.size());
      for (int l = 0; l < spectrogram.size(); l++)
      {
        new_spectrogram[l].resize(spectrogram[l].size());
      }
    }
    for (int l = 0; l < spectrogram.size(); l++)
    {
      for (int d = 0; d < spectrogram[l].size(); d++)
      {
        new_spectrogram[l][d] = spectrogram[l][d];
      }
    }
  }

  template <typename D_TYPE, typename D_TYPE2>
  void add_vector2d(std::vector<std::vector<std::complex<D_TYPE>>> &data1,
                    const std::vector<std::vector<std::complex<D_TYPE>>> &data2,
                    const D_TYPE2 coef)
  {
    for (int m = 0; m < data1.size(); m++)
    {
      for (int n = 0; n < data1[m].size(); n++)
      {
        data1[m][n] += data2[m][n] * coef;
      }
    }
  }
  template <typename D_TYPE, typename D_TYPE2>
  void minus_vector2d(std::vector<std::vector<std::complex<D_TYPE>>> &data1,
                      const std::vector<std::vector<std::complex<D_TYPE>>> &data2,
                      const D_TYPE2 coef)
  {
    for (int m = 0; m < data1.size(); m++)
    {
      for (int n = 0; n < data1[m].size(); n++)
      {
        data1[m][n] -= data2[m][n] * coef;
      }
    }
  }

  /*! @brief compute inconsistency
   * @tparam D_TYPE data type
   * @param spectrogram one spectrogram
   * @param prevSpectrogram the other spectrogram
   * @return an inconsistency measure between spectrogram and prevSpectrogram
   */
  template <typename D_TYPE>
  D_TYPE compute_inconsistency(std::vector<std::vector<std::complex<D_TYPE>>> spectrogram,
                               std::vector<std::vector<std::complex<D_TYPE>>> prevSpectrogram)
  {
    D_TYPE inconsistency = 0.0;
    D_TYPE inputPowerSum = 0.0;
    for (int m = 0; m < spectrogram.size(); m++)
    {
      for (int n = 0; n < spectrogram[m].size(); n++)
      {
        inconsistency += std::pow(std::abs(spectrogram[m][n] - prevSpectrogram[m][n]), (D_TYPE)2.0);
        inputPowerSum += std::pow(std::abs(spectrogram[m][n]), (D_TYPE)2.0);
      }
    }
    return 10 * std::log10(inconsistency / inputPowerSum);
  }

  // IO functions
  /*! @brief write spectrum
   * @tparam D_TYPE data type
   * @param spectrum complex spectrum
   * @param fp file pointer
   * @param mode output mode ('a'=ascii, 'b'=binary)
   */
  template <typename D_TYPE>
  void write_spectrum(const std::vector<std::complex<D_TYPE>> &spectrum,
                      FILE *fp,
                      const char &mode)
  {
    if (mode == 'b')
    {
      int D = spectrum.size();
      fwrite(&D, sizeof(int), 1, fp);
      fwrite(spectrum.data(), sizeof(std::complex<D_TYPE>), D, fp);
    }
    else if (mode == 'a')
    {
      for (int t = 0; t < spectrum.size(); t++)
      {
        fprintf(fp, "%f ", std::abs(spectrum[t]));
      }
      fprintf(fp, "\n");
    }
    else
    {
      throw new NotImplementedError();
    }
  }

  /*! @brief write spectrogram
   * @tparam D_TYPE data type
   * @param spectrogram complex spectrogram
   * @param fp file pointer
   * @param mode output mode ('a'=ascii, 'b'=binary)
   */
  template <typename D_TYPE>
  void write_spectrogram(const std::vector<std::vector<std::complex<D_TYPE>>> &spectrogram,
                         FILE *fp,
                         const char &mode)
  {
    if (mode == 'b')
    {
      int numOfChannels = spectrogram.size();
      fwrite(&numOfChannels, sizeof(int), 1, fp);
      for (int n = 0; n < numOfChannels; n++)
      {
        int D = spectrogram[n].size();
        fwrite(&D, sizeof(int), 1, fp);
        fwrite(spectrogram[n].data(), sizeof(std::complex<D_TYPE>), D, fp);
      }
    }
    else if (mode == 'a')
    {
      for (int n = 0; n < spectrogram.size(); n++)
      {
        for (int t = 0; t < spectrogram[n].size(); t++)
        {
          fprintf(fp, "%f ", std::abs(spectrogram[n][t]));
        }
        fprintf(fp, "\n");
      }
    }
    else
    {
      throw new NotImplementedError();
    }
  }

  /*! @brief read spectrogram
   * @tparam D_TYPE data type
   * @param spectrogram complex spectrogram
   * @param fp file pointer
   * @param mode input mode ('a'=ascii, 'b'=binary)
   */
  template <typename D_TYPE>
  void read_spectrogram(std::vector<std::vector<std::complex<D_TYPE>>> &spectrogram,
                        FILE *fp,
                        const char &mode)
  {
    if (mode == 'b')
    {
      int numOfChannels;
      fread(&numOfChannels, sizeof(int), 1, fp);
      spectrogram.resize(numOfChannels);
      for (int n = 0; n < numOfChannels; n++)
      {
        int D;
        fread(&D, sizeof(int), 1, fp);
        spectrogram[n].resize(D);
        fread(spectrogram[n].data(), sizeof(std::complex<D_TYPE>), D, fp);
      }
    }
    else
    {
      throw new NotImplementedError();
    }
  }

  /*! @brief write spectrum
   * @tparam D_TYPE data type
   * @param spectrum complex spectrum
   * @param fp file pointer
   * @param mode output mode ('a'=ascii, 'b'=binary)
   */
  template <typename D_TYPE>
  void read_spectrum(std::vector<std::complex<D_TYPE>> &spectrum,
                     FILE *fp,
                     const char &mode)
  {
    if (mode == 'b')
    {
      int D;
      fread(&D, sizeof(int), 1, fp);
      spectrum.resize(D);
      fread(spectrum.data(), sizeof(std::complex<D_TYPE>), D, fp);
    }
    else
    {
      throw new NotImplementedError();
    }
  }
} // namespace utils
