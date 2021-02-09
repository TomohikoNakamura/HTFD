/*!
 * @file cwt.hpp
 * @brief continuous wavelet transform (CWT) module containing fast approximate CWT
 * @date 3 Mar. 2015.
 * @author Tomohiko Nakamura
 */

/*! @brief This namespace contains functions and classes associated with continuous wavelet transform (CWT) and fast approximate CWT
 * @namespace cwt
 */
namespace cwt
{
  /*! @brief Approximate computation of the log-normal wavelet
   *
   * The log-normal wavelet is defined in the Fourier transform domain as
   * \f[
   *     \hat{\psi}(\omega;\sigma^2,\alpha) =
   *      \begin{cases}
   *         0 & (\omega<0) \\
   *         e^{-\frac{(\ln \omega)^2}{2\alpha\sigma^2}} & (\omega\geq 0)
   *      \end{cases}
   * \f]
   * where \f$\omega\f$ is the frequency and $\sigma$ is the standard deviation in the log-frequency domain.
   *
   * @tparam D_TYPE data type
   * @param centerAngFreqs center frequencies in normalized angular frequency
   * @param T Extended signal length up to next power of 2
   * @param P Ratio of standard deviation for approximating frequency responses of the filterbank. The frequency responses are assumed to be zero outside the range \f$[-P\sigma,P\sigma]\f$ around each center frequency.
   * @param sd Standard deviation of Gaussian shape in the log-frequency domain.
   * @param alpha If alpha=1 (2), magnitudes (power, resp.) of frequency responses have Gaussian shape.
   * @param multirateFlag If true, setup for multirate spectrogram; otherwise setup for spectrogram with common sampling rate between all channels.
   * @return a tuple of three components: indexes with which non-zero values of frequency responses start, frequency responses, sum of filter outputs.
   */
  template <typename D_TYPE>
  std::tuple<std::vector<int>, std::vector<std::vector<D_TYPE>>, std::vector<D_TYPE>>
  set_lognormal_wavelet(const std::vector<D_TYPE> &centerAngFreqs,
                        const int &T,
                        const D_TYPE &P,
                        const D_TYPE &sd,
                        const D_TYPE &alpha,
                        const D_TYPE &multirateFlag)
  {
    D_TYPE deltaAngFreq = 2.0 * M_PI / (D_TYPE)T;
    std::vector<int> startFreqIndexes(centerAngFreqs.size());
    std::vector<std::vector<D_TYPE>> freqResps(centerAngFreqs.size());
    if (P <= 0.0)
    {
      for (int l = 0; l < centerAngFreqs.size(); l++)
      {
        int startIndex = 0;
        startFreqIndexes[l] = startIndex;
        freqResps[l].resize(T / 2 + 1);
        for (int d = 0; d < T / 2 + 1; d++)
        {
          if (d == 0)
          {
            freqResps[l][d] = 0.0;
          }
          else
          {
            const D_TYPE omega = d * deltaAngFreq;
            freqResps[l][d] = std::exp(-std::pow(std::log(omega) - std::log(centerAngFreqs[l]), (D_TYPE)2.0) / (2.0 * sd * sd * alpha));
          }
        }
      }
    }
    else
    {
      if (multirateFlag)
      {
        // multirate
        for (int l = 0; l < centerAngFreqs.size(); l++)
        {
          int startIndex = std::max(1, (int)std::floor(std::exp(std::log(centerAngFreqs[l]) - P * sd) / deltaAngFreq));
          startFreqIndexes[l] = startIndex;
          int width = std::ceil((std::exp(std::log(centerAngFreqs[l]) + P * sd) - std::exp(std::log(centerAngFreqs[l]) - P * sd)) / deltaAngFreq);
          const int D = std::pow(2, std::max(1, utils::nextpow2(width)));
          //
          freqResps[l].resize(D);
          for (int d = 0; d < D; d++)
          {
            if (d + startIndex >= T / 2 + 1)
            {
              freqResps[l][d] = 0.0;
            }
            else
            {
              const D_TYPE omega = (d + startIndex) * deltaAngFreq;
              freqResps[l][d] = std::exp(-std::pow(std::log(omega) - std::log(centerAngFreqs[l]), (D_TYPE)2.0) / (2.0 * sd * sd * alpha));
            }
          }
        }
      }
      else
      {
        // single rate
        int width = 0;
        for (int l = 0; l < centerAngFreqs.size(); l++)
        {
          width = std::ceil((centerAngFreqs[l] * std::exp(P * sd) - centerAngFreqs[l] * std::exp(-P * sd)) / deltaAngFreq);
        }
        const int D = std::pow(2, std::max(1, utils::nextpow2(width)));
        for (int l = 0; l < centerAngFreqs.size(); l++)
        {
          int startIndex = std::max(1, (int)std::floor(std::exp(std::log(centerAngFreqs[l]) - P * sd) / deltaAngFreq));
          startFreqIndexes[l] = startIndex;
          freqResps[l].resize(D);
          for (int d = 0; d < D; d++)
          {
            if (d + startIndex >= T / 2 + 1)
            {
              freqResps[l][d] = 0.0;
            }
            else
            {
              const D_TYPE omega = (d + startIndex) * deltaAngFreq;
              freqResps[l][d] = std::exp(-std::pow(std::log(omega) - std::log(centerAngFreqs[l]), (D_TYPE)2.0) / (2.0 * sd * sd * alpha));
            }
          }
        }
      }
    }
    // filter output computation
    std::vector<D_TYPE> sumOfFilterOutputs(T / 2 + 1);
    for (int l = 0; l < freqResps.size(); l++)
    {
      for (int d = 0; d < freqResps[l].size(); d++)
      {
        if (startFreqIndexes[l] + d < T / 2 + 1)
        {
          sumOfFilterOutputs[startFreqIndexes[l] + d] += freqResps[l][d] * freqResps[l][d];
        }
      }
    }
    //
    return std::make_tuple(startFreqIndexes, freqResps, sumOfFilterOutputs);
  }

  template <typename D_TYPE>
  std::tuple<std::vector<int>, std::vector<std::vector<D_TYPE>>, std::vector<D_TYPE>>
  set_gabor_wavelet(const std::vector<D_TYPE> &centerAngFreqs,
                    const int &T,
                    const D_TYPE &P,
                    const D_TYPE &sd,
                    const D_TYPE &alpha,
                    const D_TYPE &multirateFlag)
  {
    D_TYPE deltaAngFreq = 2.0 * M_PI / (D_TYPE)T;
    std::vector<int> startFreqIndexes(centerAngFreqs.size());
    std::vector<std::vector<D_TYPE>> freqResps(centerAngFreqs.size());
    if (multirateFlag)
    {
      throw std::runtime_error("not implemented");
    }
    else
    {
      // single rate
      int width = 0;
      for (int l = 0; l < centerAngFreqs.size(); l++)
      {
        width = std::max(width, static_cast<int>(std::ceil((centerAngFreqs[l] * P * sd * 2) / deltaAngFreq)));
      }
      const int D = std::pow(2, std::max(1, utils::nextpow2(width)));
      for (int l = 0; l < centerAngFreqs.size(); l++)
      {
        int startIndex = std::max(1, (int)(std::floor((centerAngFreqs[l] - P * sd) / deltaAngFreq)));
        startFreqIndexes[l] = startIndex;
        freqResps[l].resize(D);
        for (int d = 0; d < D; d++)
        {
          if (d + startIndex >= T / 2 + 1)
          {
            freqResps[l][d] = 0.0;
          }
          else
          {
            const D_TYPE omega = (d + startIndex) * deltaAngFreq;
            freqResps[l][d] = std::exp(-std::pow(omega - centerAngFreqs[l], (D_TYPE)2.0) * sd * sd / (2.0 * alpha));
          }
        }
      }
    }
    // filter output computation
    std::vector<D_TYPE> sumOfFilterOutputs(T / 2 + 1);
    for (int l = 0; l < freqResps.size(); l++)
    {
      for (int d = 0; d < freqResps[l].size(); d++)
      {
        if (startFreqIndexes[l] + d < T / 2 + 1)
        {
          sumOfFilterOutputs[startFreqIndexes[l] + d] += freqResps[l][d] * freqResps[l][d];
        }
      }
    }
    //
    return std::make_tuple(startFreqIndexes, freqResps, sumOfFilterOutputs);
  }

  /*! @class CWT
   * @brief This class is for CWT and fast approximate CWT.
   * @tparam D_TYPE data type
   */
  template <typename D_TYPE>
  class CWT
  {
  protected:
    /*! @brief Set center angular frequencies
     *
     * Center frequencies are uniformly located in the log-frequency domain.
     *
     * @param lowFreq lowest frequency [Hz]
     * @param highFreq highest frequency [Hz]
     * @param fs sampling frequency [Hz]
     * @param resol Resolution of an octave
     */
    void set_centerAngFreqs(const D_TYPE lowFreq,
                            const D_TYPE highFreq,
                            const long fs,
                            const int resol)
    {
      centerAngFreqs.clear();
      D_TYPE angFreq = 2.0 * M_PI * lowFreq / fs;
      do
      {
        centerAngFreqs.push_back(angFreq);
        angFreq = angFreq * std::pow(2.0, 1.0 / (12.0 * resol));
      } while (angFreq < M_PI);
    }
    /*! @brief Save function
     * @param fp file pointer
     */
    void _save(FILE *fp)
    {
      fwrite(&sigLen, sizeof(int), 1, fp);
      fwrite(&T, sizeof(int), 1, fp);
      int numOfChannels = startFreqIndexes.size();
      fwrite(&numOfChannels, sizeof(int), 1, fp);
      fwrite(startFreqIndexes.data(), sizeof(int), numOfChannels, fp);
      fwrite(&numOfChannels, sizeof(int), 1, fp);
      for (int n = 0; n < numOfChannels; n++)
      {
        int D = freqResps[n].size();
        fwrite(&D, sizeof(int), 1, fp);
        fwrite(freqResps[n].data(), sizeof(D_TYPE), D, fp);
      }
      //
      int sumOfFilterOutputsLen = sumOfFilterOutputs.size();
      fwrite(&sumOfFilterOutputsLen, sizeof(int), 1, fp);
      fwrite(sumOfFilterOutputs.data(), sizeof(D_TYPE), sumOfFilterOutputsLen, fp);
      //
      fwrite(&numOfChannels, sizeof(int), 1, fp);
      fwrite(centerAngFreqs.data(), sizeof(D_TYPE), numOfChannels, fp);
    }
    /*! @brief Load function
     * @param fp file pointer
     */
    void _load(FILE *fp)
    {
      fread(&sigLen, sizeof(int), 1, fp);
      fread(&T, sizeof(int), 1, fp);
      //
      int numOfChannels;
      fread(&numOfChannels, sizeof(int), 1, fp);
      startFreqIndexes.resize(numOfChannels);
      fread(startFreqIndexes.data(), sizeof(int), numOfChannels, fp);
      //
      fread(&numOfChannels, sizeof(int), 1, fp);
      freqResps.resize(numOfChannels);
      for (int n = 0; n < numOfChannels; n++)
      {
        int D;
        fread(&D, sizeof(int), 1, fp);
        freqResps[n].resize(D);
        fread(freqResps[n].data(), sizeof(D_TYPE), D, fp);
      }
      //
      int sumOfFilterOutputsLen;
      fread(&sumOfFilterOutputsLen, sizeof(int), 1, fp);
      sumOfFilterOutputs.resize(sumOfFilterOutputsLen);
      fread(sumOfFilterOutputs.data(), sizeof(D_TYPE), sumOfFilterOutputsLen, fp);
      //
      fread(&numOfChannels, sizeof(int), 1, fp);
      centerAngFreqs.resize(numOfChannels);
      fread(centerAngFreqs.data(), sizeof(D_TYPE), numOfChannels, fp);
    }
    /*! @brief Initialization function
     * @param _sigLen signal length
     * @param lowFreq lowest frequency [Hz]
     * @param highFreq highest frequency [Hz]
     * @param fs sampling frequency [Hz]
     * @param resol interval number of an octave
     */
    void _init(const int _sigLen,
               const D_TYPE lowFreq,
               const D_TYPE highFreq,
               const long fs,
               const int resol)
    {
      sigLen = _sigLen;
      T = std::pow(2, utils::nextpow2(sigLen));
      set_centerAngFreqs(lowFreq, highFreq, fs, resol);
    }
    //!@brief forward FFT for signal of T length
    ForwardrealFFT frFFT;
    //!@brief inverse FFT for signal of T length
    BackwardrealFFT brFFT;
    //!@brief <T length forward and inverse FFTs
    std::vector<std::pair<ForwardFFT, BackwardFFT>> cFFTVec;
    //!@brief FFT indexes
    std::vector<int> cFFTVecIndexes;
    //!@brief fftw initialization
    void _fftw_init()
    {
      frFFT.init(T, FFTW_MEASURE);
      brFFT.init(T, FFTW_MEASURE);
    }

  public:
    //!@brief signal length
    int sigLen;
    //!@brief FFT length
    int T;
    //!@brief indexes with which non-zero values of frequency responses start
    std::vector<int> startFreqIndexes;
    //!@brief frequency responses
    std::vector<std::vector<D_TYPE>> freqResps;
    //!@brief sum of filter outputs
    std::vector<D_TYPE> sumOfFilterOutputs;
    //!@brief center frequency in normalized angular frequencies
    std::vector<D_TYPE> centerAngFreqs;
    //!@brief CWT constructor
    CWT()
    {
      sigLen = -1;
      T = -1;
    }
    //!@brief CWT destructor
    ~CWT() {}
    /*! @brief accesible initialization function
     * @param _sigLen signal length
     * @param lowFreq lowest frequency [Hz]
     * @param highFreq highest frequency [Hz]
     * @param fs sampling frequency [Hz]
     * @param resol interval number of an octave
     */
    void init(const int _sigLen,
              const D_TYPE lowFreq,
              const D_TYPE highFreq,
              const long fs,
              const int resol)
    {
      _init(_sigLen, lowFreq, highFreq, fs, resol);
    }
    //!@brief accessible initialization function of FFT instances
    void init_ffts()
    {
      _fftw_init();
      std::vector<int> sizeList;
      cFFTVecIndexes.resize(freqResps.size());
      for (int l = 0; l < freqResps.size(); l++)
      {
        const int D = freqResps[l].size();
        int pointerIndex = -1;
        for (int i = 0; i < sizeList.size(); i++)
        {
          if (sizeList[i] == D)
          {
            pointerIndex = i;
            cFFTVecIndexes[l] = pointerIndex;
            break;
          }
        }
        //
        if (pointerIndex == -1)
        {
          sizeList.push_back(D);
          cFFTVecIndexes[l] = sizeList.size() - 1;
        }
      }
      // make fft instances
      cFFTVec.resize(sizeList.size());
      for (int i = 0; i < sizeList.size(); i++)
      {
        cFFTVec[i].first.init(sizeList[i], FFTW_MEASURE);
        cFFTVec[i].second.init(sizeList[i], FFTW_MEASURE);
      }
    }
    /*! @brief Convert signal to spectrum
     * @param spectrum returned spectrum of the signal
     * @param data signal
     * @param resizeFlag If true, "spectrum" is resized.
     */
    void FFT_signal(std::vector<std::complex<D_TYPE>> &spectrum,
                    const std::vector<D_TYPE> &data,
                    const bool resizeFlag)
    {
      if (resizeFlag)
      {
        spectrum.resize(T / 2 + 1);
      }
      for (int t = 0; t < T; t++)
      {
        if (t < sigLen)
        {
          frFFT.wave[t] = data[t];
        }
        else
        {
          frFFT.wave[t] = 0.0;
        }
      }
      frFFT.perf();
      for (int t = 0; t < T / 2 + 1; t++)
      {
        spectrum[t] = frFFT.spec[t];
      }
    }
    /*! @brief Convert spectrum into signal
     * @param data returned signal
     * @param spectrum spectrum
     * @param resizeFlag If true, "data" is resized.
     */
    void IFFT_spectrum(std::vector<D_TYPE> &data,
                       const std::vector<std::complex<D_TYPE>> &spectrum,
                       const bool resizeFlag)
    {
      if (resizeFlag)
      {
        data.resize(sigLen);
      }
      for (int t = 0; t < T / 2 + 1; t++)
      {
        brFFT.spec[t] = spectrum[t];
      }
      brFFT.perf();
      for (int t = 0; t < sigLen; t++)
      {
        data[t] = brFFT.wave[t] / T;
      }
    }
    /*! @brief forward computation of CWT from spectrum of signal
     * @param spectrogram returned spectrogram of spectrum
     * @param spectrum spectrum
     * @param resizeFlag If true, spectrogram is resized.
     * @param phaseIgnoreFlag If true, circular shifting of band-limited spectra for phase is omitted.
     */
    void forward(std::vector<std::vector<std::complex<D_TYPE>>> &spectrogram,
                 const std::vector<std::complex<D_TYPE>> &spectrum,
                 const bool resizeFlag,
                 const bool phaseIgnoreFlag)
    {
      if (resizeFlag)
      {
        spectrogram.resize(freqResps.size());
        for (int l = 0; l < freqResps.size(); l++)
        {
          spectrogram[l].resize(freqResps[l].size());
        }
      }
      for (int l = 0; l < freqResps.size(); l++)
      {
        BackwardFFT *complexFFT = &(cFFTVec[cFFTVecIndexes[l]].second);
        const int B = startFreqIndexes[l];
        const int D = freqResps[l].size();
        const int n = std::ceil(B / (float)D);
        for (int d = 0; d < D; d++)
        {
          complexFFT->spec[d] = utils::C_ZERO;
        }
        if (phaseIgnoreFlag)
        {
          // ignore phase
          for (int d = 0; d < D; d++)
          {
            if (d + startFreqIndexes[l] < T / 2 + 1)
            {
              complexFFT->spec[d] = spectrum[d + B] * freqResps[l][d];
            }
            else
            {
              complexFFT->spec[d] = utils::C_ZERO;
            }
          }
        }
        else
        {
          // phase-aware
          for (int d = 0; d < std::min(B - (n - 1) * D, T / 2 + 1 - n * D); d++)
          {
            complexFFT->spec[d] = spectrum[d + n * D] * freqResps[l][d + n * D - B];
          }
          for (int d = std::max(B - (n - 1) * D, 0); d < std::min(D, T / 2 + 1 - (n - 1) * D); d++)
          {
            complexFFT->spec[d] = spectrum[d + (n - 1) * D] * freqResps[l][d + (n - 1) * D - B];
          }
        }
        complexFFT->perf();
        for (int d = 0; d < D; d++)
        {
          spectrogram[l][d] = complexFFT->wave[d] / (D_TYPE)(D);
        }
      }
    }
    /*! @brief backward computation of CWT
     * @param spectrum returned spectrum of spectrogram
     * @param spectrogram spectrogram
     * @param resizeFlag If true, spectrum is resized.
     * @param phaseIgnoreFlag If true, circular shifting of band-limited spectra for phase is omitted.
     */
    void backward(std::vector<std::complex<D_TYPE>> &spectrum,
                  const std::vector<std::vector<std::complex<D_TYPE>>> &spectrogram,
                  const bool resizeFlag,
                  const bool phaseIgnoreFlag)
    {
      if (resizeFlag)
      {
        spectrum.resize(T / 2 + 1);
      }
      for (int t = 0; t < T / 2 + 1; t++)
      {
        spectrum[t] = utils::C_ZERO;
      }
      for (int l = 0; l < freqResps.size(); l++)
      {
        ForwardFFT *complexFFT = &(cFFTVec[cFFTVecIndexes[l]].first);
        // copy
        const int B = startFreqIndexes[l];
        const int D = freqResps[l].size();
        const int n = std::ceil(B / (float)D);
        for (int d = 0; d < D; d++)
        {
          complexFFT->wave[d] = spectrogram[l][d];
        }
        complexFFT->perf();
        // circular shift
        if (phaseIgnoreFlag)
        {
          for (int d = 0; d < freqResps[l].size(); d++)
          {
            if (d + startFreqIndexes[l] < T / 2 + 1)
            {
              spectrum[d + startFreqIndexes[l]] += complexFFT->spec[d] * freqResps[l][d];
            }
          }
        }
        else
        {
          for (int d = std::max(0, B); d < std::min(n * D, T / 2 + 1); d++)
          {
            spectrum[d] += complexFFT->spec[d - (n - 1) * D] * freqResps[l][d - B];
          }
          for (int d = std::max(0, n * D); d < std::min(T / 2 + 1, B + D); d++)
          {
            spectrum[d] += complexFFT->spec[d - n * D] * freqResps[l][d - B];
          }
        }
      }
      for (int t = 0; t < T / 2 + 1; t++)
      {
        if (sumOfFilterOutputs[t] > utils::EPS)
        {
          spectrum[t] /= sumOfFilterOutputs[t];
        }
        else
        {
          spectrum[t] = utils::C_ZERO;
        }
      }
    }
    /*!@brief phase recostruction from a given magnitude spectrogram
     * @param spectrogram reconstructed spectrogram
     * @param spectrum temporal variable for computation efficiency
     * @param Iteration the number of iterations
     * @param fastCompFlag If true, replacing magnitudes is done in an inexact but fast way.
     */
    void reconstruct_phase(std::vector<std::vector<std::complex<D_TYPE>>> &spectrogram,
                           std::vector<std::complex<D_TYPE>> &spectrum,
                           const int Iteration,
                           const bool fastCompFlag)
    {
      std::vector<std::vector<D_TYPE>> magSpectrogram = utils::get_magnitude(spectrogram);
      for (int iter = 0; iter < Iteration; iter++)
      {
        //
        backward(spectrum, spectrogram, iter == 0, false);
        if (iter == Iteration - 1)
        {
          break;
        }
        forward(spectrogram, spectrum, false, false);
        // replace magnitude
        utils::replace_magnitude(spectrogram, magSpectrogram, fastCompFlag);
      }
    }
    /*! @brief save function
     * @param filename filename
     */
    void save(const char *filename)
    {
      FILE *fp = fopen(filename, "wb");
      _save(fp);
      fclose(fp);
    }
    /*! @brief load function
     * @param filename filename
     */
    void load(const char *filename)
    {
      FILE *fp = fopen(filename, "rb");
      _load(fp);
      init_ffts();
      fclose(fp);
    }
  };
}; // namespace cwt