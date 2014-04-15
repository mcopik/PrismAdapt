/**
 * 
 */
package simulator.gpu.opencl.kernel.memory;

import java.util.List;

import simulator.gpu.opencl.kernel.expression.Expression;
import simulator.gpu.opencl.kernel.expression.Include;

public interface UDType extends VariableInterface
{
	List<Include> getIncludes();

	Expression getDeclaration();

	Expression getDefinition();
}
