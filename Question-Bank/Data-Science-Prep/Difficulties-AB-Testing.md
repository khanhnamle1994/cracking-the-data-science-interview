## Problem
This problem was asked by Airbnb.

What are some factors that might make testing metrics on the Airbnb platform difficult?

## Solution
First, the booking flow is complex - it starts with a user search, often requires communication between a user and a host, or it can go directly to a booking without the host. Some of these can depend on factors outside of Airbnb’s control, such as host responsiveness. Additionally, there are different flows since sometimes a booking will be instantaneous versus a long-drawn-out process.

Secondly, users can use different devices during the booking flow or could even be not logged in for parts of the flow. If this is the case, then careful tracking needs to be done to ensure that the right data is being collected and attributed to the relevant steps of the booking flow.
