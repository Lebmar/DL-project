# ECB public data snapshot

Provider: European Central Bank. Source: [historical reference-rate ZIP](https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip). Downloaded on 2026-10-11; last observation 2026-10-09. The archive SHA-256 is `3d4f0b3877df2d995df6c6be36638b8923b283fac473e588df4d70798c32c501`.

`ecb_reference_rates.zip` is the original downloaded archive, with 7,111 publication dates from 1999-01-04. `ecb_log_returns_pct.csv` is our transformation for USD, JPY, GBP, and CHF:

```text
100 * log(reference_rate_t / previous_available_reference_rate)
```

Each rate is currency units per EUR. The first row is dropped because no previous observation exists. No weekend/holiday interpolation is applied. The prediction target is the next available publication-date return, which may span more than one calendar day. Reference rates are informational and are not executable transaction prices. These four series share EUR and are not independent markets. The downloaded history is not a point-in-time vintage archive.

ECB permits accurate reproduction with acknowledgement of the source under its [disclaimer and copyright notice](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html). The transformed return file is produced by this project and is not an ECB product or endorsement. Source attribution should remain with figures and reports.

No restricted competition data or HPC credentials are included.
