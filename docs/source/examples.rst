Examples
========
A straight forward example with random data:

.. code-block:: python

    import numpy as np
    import mne

    from pyprep.noisy import Noisydata

    # Make a random raw mne object
    sfreq = 1000.
    t = np.arange(0, 10, 1./sfreq)
    n_chans = 3
    ch_names = ['Cz', 'Pz', 'Oz']
    X = np.random.random((n_chans, t.shape[0]))
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                           ch_types=['eeg']*n_chans)
    raw = mne.io.RawArray(X, info)

    # Assign the mne object to the Noisydata class
    # All work within that class will be done on a copy
    nd = Noisydata(raw)

    # Find all bad channels and print a summary
    nd.find_all_bads()
    bads = nd.get_bads(verbose=True)

    # Now the bad channels are saved in `bads`
    # and we can continue processing our `raw` object.
    # For more information, we cann access attributes of the nd
    # instance:
    # Check the channel correlations
    nd._channel_correlations

    # Check the high frequency noise per channel
    nd._channel_hf_noise

    # and so on ...

For finding bad epochs, it is recommended to highpass the data or apply
baseline correction. Furthermore, bad channels should be identified and removed
or interpolated beforehand.

.. code-block:: python
    import numpy as np
    import mne

    from pyprep.noisy importfind_bad_epochs

    # You can also define picks, and the threshold to be used
    bads = find_bad_epochs(epochs, picks=None, thresh=3.29053)
