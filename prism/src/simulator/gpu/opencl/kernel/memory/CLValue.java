/**
 * 
 */
package simulator.gpu.opencl.kernel.memory;

import simulator.gpu.opencl.kernel.expression.Expression;

public interface CLValue
{
	boolean validateAssignmentTo(VariableInterface type);

	Expression getSource();
}
